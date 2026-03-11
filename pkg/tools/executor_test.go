package tools

import (
	"encoding/json"
	"testing"

	"github.com/matst80/go-ollama-client/pkg/ai"
)

// These tests exercise the various code paths in ToolExecutor/HandleCalls that
// were previously not covered by the session-coupled test.

func TestToolExecutor_NoReturn(t *testing.T) {
	registry := NewRegistry()

	// Handler with no return values should result in content "ok".
	handler := func(args TestArgs) {
		// intentionally no return
	}
	if err := registry.Register("no_return", &TestArgs{}, handler); err != nil {
		t.Fatalf("failed to register tool: %v", err)
	}

	executor := NewToolExecutor(registry)

	calls := []ai.ToolCall{
		{
			ID: "nr_1",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "no_return",
				Arguments: json.RawMessage(`{"command":"x"}`),
			},
		},
	}

	results, err := executor.HandleCalls(calls)
	if err != nil {
		t.Fatalf("HandleCalls returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	r := results[0]
	if r.Err != nil {
		t.Fatalf("unexpected error for no-return handler: %v", r.Err)
	}
	if r.Content != "ok" {
		t.Fatalf("expected content 'ok' for no-return handler, got '%s'", r.Content)
	}
	if !executor.IsHandled("nr_1") {
		t.Fatalf("expected call to be marked handled")
	}
}

func TestToolExecutor_UnmarshalError(t *testing.T) {
	registry := NewRegistry()

	// Handler which expects a bool for Wait; we'll pass an incorrect JSON type
	handler := func(args TestArgs) string {
		return "should not get here"
	}
	if err := registry.Register("expect_bool", &TestArgs{}, handler); err != nil {
		t.Fatalf("failed to register tool: %v", err)
	}

	executor := NewToolExecutor(registry)

	// Provide invalid JSON for the struct (wait expects a bool; provide string)
	calls := []ai.ToolCall{
		{
			ID: "bad_unmarshal",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "expect_bool",
				Arguments: json.RawMessage(`{"command":"ls","wait":"notabool"}`),
			},
		},
	}

	results, err := executor.HandleCalls(calls)
	if err != nil {
		t.Fatalf("HandleCalls returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	r := results[0]
	// We expect an error due to failed unmarshal
	if r.Err == nil {
		t.Fatalf("expected unmarshal error, got nil")
	}
	// Content should be empty when there is an execution error
	if r.Content != "" {
		t.Fatalf("expected empty content on error, got '%s'", r.Content)
	}
	// Call should be marked handled (Call sets handled flag even on error)
	if !executor.IsHandled("bad_unmarshal") {
		t.Fatalf("expected call to be marked handled despite error")
	}
}

func TestToolExecutor_DuplicateAndEmptyID(t *testing.T) {
	registry := NewRegistry()

	handler := func(args TestArgs) string {
		return "dup-handled"
	}
	if err := registry.Register("dup_tool", &TestArgs{}, handler); err != nil {
		t.Fatalf("failed to register tool: %v", err)
	}

	executor := NewToolExecutor(registry)

	calls := []ai.ToolCall{
		// empty ID should be skipped entirely
		{
			ID: "",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "dup_tool",
				Arguments: json.RawMessage(`{"command":"skipme"}`),
			},
		},
		// duplicate ID - two calls with same ID; second should be ignored
		{
			ID: "dup1",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "dup_tool",
				Arguments: json.RawMessage(`{"command":"first"}`),
			},
		},
		{
			ID: "dup1",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "dup_tool",
				Arguments: json.RawMessage(`{"command":"second"}`),
			},
		},
	}

	results, err := executor.HandleCalls(calls)
	if err != nil {
		t.Fatalf("HandleCalls returned error: %v", err)
	}
	// Expect one result (the duplicate pair yields a single execution)
	if len(results) != 1 {
		t.Fatalf("expected 1 result for duplicate IDs, got %d", len(results))
	}
	if results[0].CallID != "dup1" {
		t.Fatalf("expected result CallID 'dup1', got '%s'", results[0].CallID)
	}
	if results[0].Content != "dup-handled" {
		t.Fatalf("unexpected content for duplicate call: %s", results[0].Content)
	}
	// empty ID should never be marked handled
	if executor.IsHandled("") {
		t.Fatalf("empty ID should not be marked handled")
	}
	// dup1 should be marked handled
	if !executor.IsHandled("dup1") {
		t.Fatalf("expected dup1 to be marked handled")
	}
}

func TestToolExecutor_UnknownToolNotIgnored(t *testing.T) {
	registry := NewRegistry()

	// Do not register the tool; we want unknown-tool path
	executor := NewToolExecutor(registry) // default: IgnoreUnknownToolCalls == false

	calls := []ai.ToolCall{
		{
			ID: "unknown1",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "i_do_not_exist",
				Arguments: json.RawMessage(`{}`),
			},
		},
	}

	results, err := executor.HandleCalls(calls)
	if err != nil {
		t.Fatalf("HandleCalls returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result for unknown tool, got %d", len(results))
	}
	r := results[0]
	if r.Err == nil {
		t.Fatalf("expected an error for unknown tool when not ignoring, got nil")
	}
	// The executor should not mark the unknown call as handled in this branch
	if executor.IsHandled("unknown1") {
		t.Fatalf("expected unknown1 NOT to be marked handled when unknown tools are not ignored")
	}
}

func TestToolExecutor_MultipleReturnValues(t *testing.T) {
	registry := NewRegistry()

	// Handler returns multiple values: string and int
	handler := func(args TestArgs) (string, int) {
		return "multi", 42
	}
	if err := registry.Register("multi_tool", &TestArgs{}, handler); err != nil {
		t.Fatalf("failed to register tool: %v", err)
	}

	executor := NewToolExecutor(registry)

	calls := []ai.ToolCall{
		{
			ID: "multi1",
			Function: struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}{
				Name:      "multi_tool",
				Arguments: json.RawMessage(`{"command":"anything"}`),
			},
		},
	}

	results, err := executor.HandleCalls(calls)
	if err != nil {
		t.Fatalf("HandleCalls returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result for multi return, got %d", len(results))
	}
	r := results[0]
	if r.Err != nil {
		t.Fatalf("unexpected error for multi return: %v", r.Err)
	}
	// Expect JSON array encoding of the return values
	expected := `["multi",42]`
	if r.Content != expected {
		t.Fatalf("expected multi-return content %s, got %s", expected, r.Content)
	}
}
