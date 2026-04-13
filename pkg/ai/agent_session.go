package ai

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

type AgentSessionOption func(*AgentSession)

type AutoToolResult struct {
	CallID  string
	Content string
	Err     error
}

type AgentSessionInterface interface {
	SendUserMessage(ctx context.Context, msg string) error
	SendMessages(ctx context.Context, msgs ...Message) error
	Recv() <-chan AccumulatedResponse
	GetMessageHistory() []Message
	GetModel() string
	Stop()
	GetState() AgentState
	SetState(update func(AgentState))
	SetTools(tools []Tool)
	SetClient(client ChatClientInterface)
	SetHooks(hooks ...SessionHooks)
	GetContext() context.Context
}

type AgentStatus string

const (
	AgentStatusIdle    AgentStatus = "idle"
	AgentStatusRunning AgentStatus = "running"
	AgentStatusStopped AgentStatus = "stopped"
)

// AgentState holds the current status and metadata of an agent session
type AgentState interface {
	SetStatus(AgentStatus)
	GetStatus() AgentStatus
	SetParentID(string)
	GetParentID() string
	SetTitle(string)
	GetTitle() string
	SetType(string)
	GetType() string
	SetCreatedAt(time.Time)
	GetCreatedAt() time.Time
	SetLastActive(time.Time)
	GetLastActive() time.Time
}

type DefaultAgentState struct {
	status     AgentStatus
	parentID   string
	title      string
	agentType  string
	createdAt  time.Time
	lastActive time.Time
}

func (s *DefaultAgentState) SetStatus(status AgentStatus) {
	s.status = status
}

func (s *DefaultAgentState) GetStatus() AgentStatus {
	return s.status
}

func (s *DefaultAgentState) SetParentID(parentID string) {
	s.parentID = parentID
}

func (s *DefaultAgentState) GetParentID() string {
	return s.parentID
}

func (s *DefaultAgentState) SetTitle(title string) {
	s.title = title
}

func (s *DefaultAgentState) GetTitle() string {
	return s.title
}

func (s *DefaultAgentState) SetType(agentType string) {
	s.agentType = agentType
}

func (s *DefaultAgentState) GetType() string {
	return s.agentType
}

func (s *DefaultAgentState) SetCreatedAt(createdAt time.Time) {
	s.createdAt = createdAt
}

func (s *DefaultAgentState) GetCreatedAt() time.Time {
	return s.createdAt
}

func (s *DefaultAgentState) SetLastActive(lastActive time.Time) {
	s.lastActive = lastActive
}

func (s *DefaultAgentState) GetLastActive() time.Time {
	return s.lastActive
}

func NewDefaultAgentState() *DefaultAgentState {
	return &DefaultAgentState{
		status:     AgentStatusIdle,
		createdAt:  time.Now(),
		lastActive: time.Now(),
	}
}

// AgentSession manages a continuous chat session, tracking the underlying ChatRequest
// and providing a single global channel for responses of type T.
type AgentSession struct {
	client           ChatClientInterface
	rec              *ChatRequest
	globalChan       chan AccumulatedResponse
	ctx              context.Context
	cancel           context.CancelFunc
	mu               sync.Mutex
	state            AgentState
	truncationConfig *TruncationConfig
	// optional repository root used by the diff parser
	repoRoot string
	// optional operation handler used by the diff parser
	opHandler OperationHandler
	// internal parser instance (set when a stream starts)
	diffParser *DiffParser
	stopOnce   sync.Once

	autoToolHandler func(context.Context, []ToolCall) ([]Message, []AutoToolResult, error)
	OnChatRequest   func(context.Context, *ChatRequest) error
	hooks           []SessionHooks
}

// TruncationConfig holds optional truncation settings for a session.
type TruncationConfig struct {
	Strategy TruncationStrategy
}

// NewAgentSession creates a new AgentSession with the given client and base request.
// Use options like WithTransformer or WithAccumulator to specify the output type.
func NewAgentSession(ctx context.Context, client ChatClientInterface, req *ChatRequest, state AgentState, opts ...AgentSessionOption) *AgentSession {
	ctx, cancel := context.WithCancel(ctx)
	session := &AgentSession{
		client: client,
		rec:    req,
		// We can't pre-allocate GlobalChan with T if we don't know the capacity needed,
		// but 100 is a safe default for streaming.
		globalChan: make(chan AccumulatedResponse, 100),
		ctx:        ctx,
		cancel:     cancel,
		state:      state,
	}
	for _, opt := range opts {
		opt(session)
	}
	return session
}

// WithRepoRoot configures a repo root path used by the diff parser.
func WithRepoRoot(path string) AgentSessionOption {
	return func(a *AgentSession) {
		a.repoRoot = path
	}
}

// WithOperationHandler sets a composable OperationHandler for streamed diff operations.
func WithOperationHandler(h OperationHandler) AgentSessionOption {
	return func(a *AgentSession) {
		a.opHandler = h
	}
}

// WithTruncation returns an AgentSessionOption that configures a truncation strategy.
func WithTruncation(strategy TruncationStrategy) AgentSessionOption {
	return func(a *AgentSession) {
		a.truncationConfig = &TruncationConfig{Strategy: strategy}
	}
}

// WithAutoToolExecutor configures the session to automatically execute tool calls
// emitted at the end of a response and feed the tool result messages back into the
// same session. The callback is optional and is invoked once for each tool result.
func WithAutoToolExecutor(
	handler func(context.Context, []ToolCall) ([]Message, []AutoToolResult, error),
) AgentSessionOption {
	return func(a *AgentSession) {
		a.autoToolHandler = handler
	}
}

// WithOnChatRequest returns an AgentSessionOption that configures an OnChatRequest hook.
func WithOnChatRequest(hook func(context.Context, *ChatRequest) error) AgentSessionOption {
	return func(a *AgentSession) {
		a.OnChatRequest = hook
	}
}

// WithHooks returns an AgentSessionOption that registers one or more session hooks.
func WithHooks(hooks ...SessionHooks) AgentSessionOption {
	return func(a *AgentSession) {
		a.hooks = append(a.hooks, hooks...)
	}
}

// WithMemory returns an AgentSessionOption that configures a memory store.
// It automatically adds an AutomaticMemoryHook to retrieve context and,
// if a truncation strategy is already configured, wraps it in a MemoryTruncator.
func WithMemory(store MemoryStore, maxMemories int) AgentSessionOption {
	return func(a *AgentSession) {
		// Add the retrieval hook
		a.hooks = append(a.hooks, NewAutomaticMemoryHook(store, maxMemories))

		// If we have a strategy, wrap it so truncated messages are stored
		if a.truncationConfig != nil && a.truncationConfig.Strategy != nil {
			a.truncationConfig.Strategy = NewMemoryTruncator(a.truncationConfig.Strategy, store, nil)
		}
	}
}

// Recv returns a receive-only channel for all chat responses across multiple turns.
func (a *AgentSession) Recv() <-chan AccumulatedResponse {
	return a.globalChan
}

// Stop cancels the session context and closes the global channel.
func (a *AgentSession) Stop() {
	a.stopOnce.Do(func() {
		a.cancel()
		close(a.globalChan)
	})
}

func (a *AgentSession) GetContext() context.Context {
	return a.ctx
}

// SendUserMessage appends a user message to the request and triggers a streaming chat.
// It serializes request mutation and stream startup under the session mutex.
func (a *AgentSession) SendUserMessage(ctx context.Context, msg string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	// If the last message is already a User message with the same content,
	// just trigger the response without adding it again.
	if lastIdx := len(a.rec.Messages) - 1; lastIdx >= 0 {
		last := a.rec.Messages[lastIdx]
		if last.Role == MessageRoleUser && last.Content == msg {
			return a.streamChat(ctx)
		}
	}

	a.rec.AddMessage(MessageRoleUser, msg)
	// Apply truncation if configured (must run while locked)
	a.applyTruncationLocked()
	return a.streamChat(ctx)
}

// SendMessages appends one or more pre-constructed messages (like tool results) and triggers a streaming chat.
// It serializes request mutation and stream startup under the session mutex.
func (a *AgentSession) SendMessages(ctx context.Context, msgs ...Message) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.rec.Messages = append(a.rec.Messages, msgs...)
	// Apply truncation if configured (must run while locked)
	a.applyTruncationLocked()
	return a.streamChat(ctx)
}

func (a *AgentSession) SetHooks(hooks ...SessionHooks) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.hooks = hooks
}

// applyTruncationLocked applies the configured truncation strategy to a.rec.Messages.
// Caller must hold a.mu.
func (a *AgentSession) applyTruncationLocked() {
	if a.truncationConfig == nil || a.truncationConfig.Strategy == nil {
		return
	}

	// Make a copy for strategy to inspect
	msgs := a.rec.Messages
	truncated, removed := a.truncationConfig.Strategy.Apply(msgs)
	if removed <= 0 {
		return
	}

	// Replace messages with truncated result
	a.rec.Messages = truncated
}

// GetMessageHistory returns the current conversation history.
func (a *AgentSession) GetMessageHistory() []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.rec.Messages
}

func (a *AgentSession) SetSystemMessage(msg string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.rec.Messages) > 0 && a.rec.Messages[0].Role == MessageRoleSystem {
		a.rec.Messages[0].Content = msg
	} else {
		a.rec.Messages = append([]Message{{Role: MessageRoleSystem, Content: msg, CreatedAt: time.Now()}}, a.rec.Messages...)
	}
}

func (a *AgentSession) GetModel() string {
	return a.rec.Model
}

func (a *AgentSession) GetState() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

func (a *AgentSession) SetState(update func(AgentState)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	update(a.state)
}

// SetTools updates the tools available to the AI for subsequent requests.
func (a *AgentSession) SetTools(tools []Tool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.rec.Tools = tools
}

// SetClient updates the underlying client used by the session.
func (a *AgentSession) SetClient(client ChatClientInterface) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.client = client
}

// streamChat performs the streamed chat request and pipes the results into GlobalChan.
func (a *AgentSession) streamChat(ctx context.Context) error {
	// Ensure agent identity is preserved in the context
	if ctx.Value("agentID") == nil && a.ctx.Value("agentID") != nil {
		ctx = context.WithValue(ctx, "agentID", a.ctx.Value("agentID"))
	}
	if ctx.Value("parentAgentID") == nil && a.ctx.Value("parentAgentID") != nil {
		ctx = context.WithValue(ctx, "parentAgentID", a.ctx.Value("parentAgentID"))
	}

	if a.OnChatRequest != nil {
		if err := a.OnChatRequest(ctx, a.rec); err != nil {
			return fmt.Errorf("OnChatRequest hook failed: %w", err)
		}
	}
	for _, h := range a.hooks {
		if err := h.OnChatRequest(ctx, a.rec); err != nil {
			return fmt.Errorf("Session hook OnChatRequest failed: %w", err)
		}
	}
	ch := make(chan *ChatResponse)

	// Start the request in a goroutine with retry logic
	go func() {
		defer close(ch)

		maxRetries := 3
		var lastErr error
		for i := 0; i < maxRetries; i++ {
			if i > 0 {
				fmt.Printf("Retrying ChatStreamed (%d/%d)... error was: %v\n", i, maxRetries, lastErr)
			}

			// Create a temporary channel for this attempt
			tempCh := make(chan *ChatResponse)

			// We need a separate goroutine because ChatStreamed is blocking
			go func() {
				a.state.SetStatus("waiting")
				a.state.SetLastActive(time.Now())
				defer a.state.SetStatus("idle")
				defer a.state.SetLastActive(time.Now())
				err := a.client.ChatStreamed(ctx, *a.rec, tempCh)
				if err != nil {
					lastErr = err
				}
			}()

			// Pipe tempCh to ch
			success := false
			for res := range tempCh {
				ch <- res
				success = true
			}

			// If success is true, we already sent chunks to the consumer.
			// We MUST NOT retry because the accumulator already has these chunks.
			if success || lastErr == nil {
				return
			}

			// If context is cancelled, don't retry
			if ctx.Err() != nil {
				break
			}
		}

		if lastErr != nil {
			errStr := lastErr.Error()
			fmt.Printf("ChatStreamed failed after %d retries: %v\n", maxRetries, lastErr)
			ch <- &ChatResponse{
				BaseResponse: &BaseResponse{
					Error: &errStr,
					Done:  true,
				},
			}
		}
	}()

	// Create a DiffParser and attach it to the accumulated stream so parsed
	// fenced blocks can be handled as they arrive. Replace the repoRoot below
	// with your workspace/repo path or wire it via an AgentSession option.
	repo := "."
	if a.repoRoot != "" {
		repo = a.repoRoot
	}
	parser := NewDiffParser(repo)
	if a.opHandler != nil {
		parser.SetHandler(a.opHandler)
	}
	// store parser on session for later inspection (e.g., PopReports)
	a.diffParser = parser

	// Build a list of block handlers
	var handlers []BlockHandler
	handlers = append(handlers, NewGitDiffBlockHandler(parser))
	if len(a.hooks) > 0 {
		handlers = append(handlers, &hooksBlockHandler{hooks: a.hooks})
	}

	// Use StreamAccumulator to get accumulated responses and attach the
	// generic fenced block parser so exact fenced git diff blocks are parsed
	// before other consumers see the accumulated messages.
	accCh := AttachBlockParserToAccumulator(ctx, StreamAccumulator(ctx, ch, false), NewFenceParser(), multiBlockHandler(handlers))
	go func() {
		defer func() {
			if r := recover(); r != nil {
				// Handle panic on closed channel or other issues gracefully
				fmt.Printf("Recovered in streamChat: %v\n", r)
			}
		}()
		var last *AccumulatedResponse
		for res := range accCh {
			last = res

			// Call thinking and content hooks if we have a chunk with deltas
			if res.Chunk != nil && res.Chunk.Message.ReasoningContent != "" {
				for _, h := range a.hooks {
					h.OnThinking(ctx, res.Chunk.Message.ReasoningContent)
				}
			}
			if res.Chunk != nil && res.Chunk.Message.Content != "" {
				for _, h := range a.hooks {
					h.OnContent(ctx, res.Chunk.Message.Content)
				}
			}

			select {
			case a.globalChan <- *res:
			case <-a.ctx.Done():
				return
			case <-ctx.Done():
				return
			}
		}

		// When the stream is finished, append the assistant's final accumulated response to history
		// but only if it was successful (not an error and has content or tool calls)
		if last != nil && (last.Chunk == nil || last.Chunk.Error == nil) {
			a.mu.Lock()
			a.rec.Messages = append(a.rec.Messages, Message{
				Role:             MessageRoleAssistant,
				Content:          last.Content,
				ReasoningContent: last.ReasoningContent,
				ToolCalls:        last.ToolCalls,
				CreatedAt:        time.Now(),
			})
			a.mu.Unlock()

			if last.Chunk != nil && last.Chunk.Done && len(last.ToolCalls) > 0 && a.autoToolHandler != nil {
				// Call OnBeforeToolCall hooks
				for _, h := range a.hooks {
					if err := h.OnBeforeToolCall(ctx, last.ToolCalls); err != nil {
						fmt.Printf("Session hook OnBeforeToolCall failed: %v\n", err)
						// Continue with tool execution or abort?
						// For now, only abort if the client wants us to (later we might add return handle)
					}
				}

				resultMsgs, results, err := a.autoToolHandler(ctx, last.ToolCalls)
				if err != nil {
					fmt.Printf("Tool execution error: %v\n", err)
				}

				// Call OnAfterToolCall hooks
				for _, h := range a.hooks {
					h.OnAfterToolCall(ctx, last.ToolCalls, resultMsgs, results)
				}

				if len(resultMsgs) > 0 {
					if err := a.SendMessages(ctx, resultMsgs...); err != nil {
						fmt.Printf("failed to deliver tool results: %v\n", err)
					}
				}
			}

			// If the diff parser produced operation reports, emit a summary message back
			if a.diffParser != nil {
				reports := a.diffParser.PopReports()
				if len(reports) > 0 {
					// Build a simple textual summary
					var sb strings.Builder
					sb.WriteString("[diff-report]\n")
					for _, r := range reports {
						status := "OK"
						if !r.Success {
							status = "FAILED"
						}
						sb.WriteString(fmt.Sprintf("%s %s %s\n", status, r.Op, r.Path))
						if r.Message != "" {
							sb.WriteString("  -> " + r.Message + "\n")
						}
					}

					// send as one final accumulated response so consumers see it
					rep := AccumulatedResponse{
						Chunk:   &ChatResponse{BaseResponse: &BaseResponse{Done: true}},
						Content: sb.String(),
					}
					for _, h := range a.hooks {
						h.OnContent(ctx, rep.Content)
					}

					hasFailure := false
					for _, r := range reports {
						if !r.Success {
							hasFailure = true
							break
						}
					}

					// Feed back to AI history so it knows what happened
					a.mu.Lock()
					a.rec.Messages = append(a.rec.Messages, Message{
						Role:      MessageRoleUser,
						Content:   sb.String(),
						CreatedAt: time.Now(),
					})
					a.mu.Unlock()

					if hasFailure {
						go a.streamChat(ctx)
					}

					select {
					case a.globalChan <- rep:
					case <-a.ctx.Done():
						return
					case <-ctx.Done():
						return
					default:
					}
				}
			}
		}

		// Notify OnDone hook
		if last != nil {
			for _, h := range a.hooks {
				h.OnDone(ctx, *last)
			}
		}

		// Handle error hooks if terminal was reached with an error
		if last != nil && last.Chunk != nil && last.Chunk.Error != nil {
			errStr := *last.Chunk.Error
			for _, h := range a.hooks {
				h.OnError(ctx, fmt.Errorf("%s", errStr))
			}
		}
	}()

	return nil
}

type hooksBlockHandler struct {
	hooks []SessionHooks
}

func (h *hooksBlockHandler) HandleBlock(ctx context.Context, block *StreamedBlock) error {
	for _, hook := range h.hooks {
		hook.OnBlock(ctx, block.Type, block.Content)
	}
	return nil
}

type multiBlockHandler []BlockHandler

func (m multiBlockHandler) HandleBlock(ctx context.Context, block *StreamedBlock) error {
	var firstErr error
	for _, h := range m {
		if err := h.HandleBlock(ctx, block); err != nil {
			if firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}
