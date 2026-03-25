package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// ToolExecutor handles execution of tool calls and delivers the results into a
// specific Ollama session. The executor depends on an abstract Client
// interface (instead of the concrete client type) to make testing/mocking easier.
type ToolExecutor struct {
	registry         *Registry
	handledToolCalls map[string]bool
	// When true, unknown tool calls (registry returns "not found") will be ignored
	// instead of returning an error. This is useful for agents that emit built-in
	// tools the runtime does not implement.
	IgnoreUnknownToolCalls bool
	OnProgress             func(call ai.ToolCall, elapsed time.Duration)
}

type progressKey struct{}

// WithProgress returns a context that carries a tool progress callback
func WithProgress(ctx context.Context, fn func(ai.ToolCall, time.Duration)) context.Context {
	return context.WithValue(ctx, progressKey{}, fn)
}

func getProgress(ctx context.Context) func(ai.ToolCall, time.Duration) {
	if fn, ok := ctx.Value(progressKey{}).(func(ai.ToolCall, time.Duration)); ok {
		return fn
	}
	return nil
}

// NewToolExecutor creates a new ToolExecutor wired to an Ollama client interface and session ID.
// The executor will deliver tool responses into the provided session. Accepting the interface
// allows tests to inject mocks.
//
// The constructor supports functional options to configure behavior such as whether
// unknown tool calls should be ignored.
type ToolExecutorOption func(*ToolExecutor)

// WithIgnoreUnknownToolCalls sets the executor to ignore calls to unknown tools
// (i.e. registry.Call returns a \"not found\" error) instead of returning an
// error from HandleCalls.
func WithIgnoreUnknownToolCalls(ignore bool) ToolExecutorOption {
	return func(e *ToolExecutor) {
		e.IgnoreUnknownToolCalls = ignore
	}
}

func NewToolExecutor(registry *Registry, opts ...ToolExecutorOption) *ToolExecutor {
	e := &ToolExecutor{
		registry:         registry,
		handledToolCalls: make(map[string]bool, 0),
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

func (e *ToolExecutor) IsHandled(callID string) bool {
	_, ok := e.handledToolCalls[callID]
	return ok
}

// Call executes a single tool call and attempts to deliver the result into the configured session.
// It returns the textual result (or error string) and an error if execution or delivery failed.
//
// Behavior notes:
// - If the call has no ID, an error is returned because responses cannot be correlated.
// - Duplicate calls (same ID) are ignored and return an empty string with nil error.
// - If delivery into the session fails, the result is still returned along with the delivery error.
func (e *ToolExecutor) Call(ctx context.Context, call ai.ToolCall) (ToolResult, error) {
	if call.ID != "" {
		e.handledToolCalls[call.ID] = true
	}

	// Get definition for timeout
	timeout := 30 * time.Second
	if def, ok := e.registry.GetTool(call.Function.Name); ok && def.Timeout > 0 {
		timeout = def.Timeout
	}

	type callResult struct {
		values []reflect.Value
		err    error
	}
	resChan := make(chan callResult, 1)

	go func() {
		results, err := e.registry.Call(ctx, call.Function.Name, call.Function.Arguments)
		resChan <- callResult{results, err}
	}()

	start := time.Now()
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case res := <-resChan:
			if res.err != nil {
				return ToolResult{}, fmt.Errorf("tool execution error: %w", res.err)
			}
			content, images := processResults(res.values)
			return ToolResult{
				CallID:  call.ID,
				Content: content,
				Images:  images,
			}, nil
		case <-ticker.C:
			elapsed := time.Since(start).Round(time.Second)
			slog.Debug("Tool still executing",
				"tool", call.Function.Name,
				"call_id", call.ID,
				"elapsed", elapsed,
			)
			if fn := getProgress(ctx); fn != nil {
				fn(call, elapsed)
			} else if e.OnProgress != nil {
				e.OnProgress(call, elapsed)
			}
		case <-time.After(timeout - time.Since(start)):
			return ToolResult{}, fmt.Errorf("tool execution timed out after %v", timeout)
		case <-ctx.Done():
			return ToolResult{}, ctx.Err()
		}
	}
}

type ToolResult struct {
	CallID  string
	Content string
	Images  []string
	Err     error
}

func (r ToolResult) ToResultMessage() *ai.Message {
	if r.Err != nil {
		return &ai.Message{
			Role:       ai.MessageRoleTool,
			ToolCallID: r.CallID,
			Content:    fmt.Sprintf("error: %v", r.Err),
		}
	}
	return &ai.Message{
		Role:       ai.MessageRoleTool,
		ToolCallID: r.CallID,
		Content:    r.Content,
		Images:     r.Images,
	}
}

// HandleCalls executes multiple calls by delegating to Call for each call.
// It returns the first non-nil error encountered (typically delivery errors).
func (e *ToolExecutor) HandleCalls(ctx context.Context, calls []ai.ToolCall) ([]ToolResult, error) {

	res := make([]ToolResult, 0)
	for _, c := range calls {
		// skip calls without an ID since we can't correlate responses
		if c.ID == "" {
			continue
		}
		if e.IsHandled(c.ID) {
			continue
		}
		if !e.registry.HasTool(c.Function.Name) {
			if e.IgnoreUnknownToolCalls {
				// Mark as handled to avoid repeated attempts on future calls with the same ID
				e.handledToolCalls[c.ID] = true
				continue
			} else {
				res = append(res, ToolResult{
					CallID:  c.ID,
					Content: "",
					Err:     fmt.Errorf("unknown tool: %s", c.Function.Name),
				})
				continue
			}
		}

		tr, err := e.Call(ctx, c)
		if err != nil {
			tr.Err = err
		}
		res = append(res, tr)
	}
	return res, nil
}

func (e *ToolExecutor) AgentSessionOption(onResult func(ToolResult)) ai.AgentSessionOption {
	return ai.WithAutoToolExecutor(func(ctx context.Context, calls []ai.ToolCall) ([]ai.Message, []ai.AutoToolResult, error) {
		results, err := e.HandleCalls(ctx, calls)
		if err != nil {
			return nil, nil, err
		}

		resultMsgs := make([]ai.Message, 0, len(results))
		autoResults := make([]ai.AutoToolResult, 0, len(results))

		for _, tr := range results {
			if onResult != nil {
				onResult(tr)
			}

			msg := tr.ToResultMessage()
			resultMsgs = append(resultMsgs, *msg)
			autoResults = append(autoResults, ai.AutoToolResult{
				CallID:  tr.CallID,
				Content: tr.Content,
				Err:     tr.Err,
			})
		}

		return resultMsgs, autoResults, nil
	})
}

func processResults(results []reflect.Value) (string, []string) {
	if len(results) == 0 {
		return "ok", nil
	} else if len(results) == 1 {
		val := results[0].Interface()
		if mm, ok := val.(ai.MultimodalToolResult); ok {
			return mm.Content, mm.Images
		}
		if mm, ok := val.(*ai.MultimodalToolResult); ok {
			return mm.Content, mm.Images
		}
		return fmt.Sprint(val), nil
	} else {
		// multiple return values, encode as JSON array
		var interfaces []any
		for _, v := range results {
			interfaces = append(interfaces, v.Interface())
		}
		data, _ := json.Marshal(interfaces)
		return string(data), nil
	}
}
