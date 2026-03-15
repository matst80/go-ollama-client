package ai

import (
	"context"
	"encoding/json"
	"testing"
)

func asPrt[T any](v T) *T {
	return &v
}

func TestStreamAccumulator(t *testing.T) {
	ctx := context.Background()
	input := make(chan *ChatResponse)

	// Start transformer
	output := StreamAccumulator(ctx, input, false)

	// Feed chunks
	chunks := []*ChatResponse{
		{
			Message: Message{Content: "Hello "},
		},
		{
			Message: Message{ReasoningContent: "Thinking... "},
		},
		{
			Message: Message{Content: "world!"},
		},
		{
			Message: Message{ReasoningContent: "Done."},
		},
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: asPrt(0),
						ID:    "call_1",
						Type:  "function",
						Function: struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						}{
							Name:      "get_weather",
							Arguments: json.RawMessage(`{"city":`),
						},
					},
				},
			},
		},
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: asPrt(0),
						ID:    "call_1",
						Type:  "function",
						Function: struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						}{
							Arguments: json.RawMessage(`"Oslo"}`),
						},
					},
				},
			},
		},
	}

	go func() {
		for _, c := range chunks {
			input <- c
		}
		close(input)
	}()

	var lastAcc *AccumulatedResponse
	for acc := range output {
		lastAcc = acc
	}

	if lastAcc == nil {
		t.Fatal("Expected at least one response")
	}

	// Verify Content
	expectedContent := "Hello world!"
	if lastAcc.Content != expectedContent {
		t.Errorf("Expected content %q, got %q", expectedContent, lastAcc.Content)
	}

	// Verify ReasoningContent
	expectedReasoning := "Thinking... Done."
	if lastAcc.ReasoningContent != expectedReasoning {
		t.Errorf("Expected reasoning content %q, got %q", expectedReasoning, lastAcc.ReasoningContent)
	}

	// Verify ToolCalls
	if len(lastAcc.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(lastAcc.ToolCalls))
	}
	tc := lastAcc.ToolCalls[0]
	if string(tc.Function.Arguments) != `{"city":"Oslo"}` {
		t.Errorf("Expected tool call arguments %q, got %q", `{"city":"Oslo"}`, string(tc.Function.Arguments))
	}
}

func TestStreamAccumulator_ToolCallIndices(t *testing.T) {
	ctx := context.Background()
	input := make(chan *ChatResponse)
	output := StreamAccumulator(ctx, input, false)

	idx0 := 0
	chunks := []*ChatResponse{
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: &idx0,
						ID:    "call_123",
						Type:  "function",
						Function: FunctionCall{
							Name:      "run",
							Arguments: json.RawMessage(""),
						},
					},
				},
			},
		},
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: &idx0,
						Function: FunctionCall{
							Arguments: json.RawMessage(`{"command": `),
						},
					},
				},
			},
		},
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: &idx0,
						Function: FunctionCall{
							Arguments: json.RawMessage(`"df -h"}`),
						},
					},
				},
			},
		},
	}

	go func() {
		for _, c := range chunks {
			input <- c
		}
		close(input)
	}()

	var lastAcc *AccumulatedResponse
	for acc := range output {
		lastAcc = acc
	}

	if len(lastAcc.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(lastAcc.ToolCalls))
	}

	tc := lastAcc.ToolCalls[0]
	if tc.ID != "call_123" {
		t.Errorf("Expected ID 'call_123', got %s", tc.ID)
	}
	if tc.Function.Name != "run" {
		t.Errorf("Expected name 'run', got %s", tc.Function.Name)
	}
	if string(tc.Function.Arguments) != `{"command": "df -h"}` {
		t.Errorf("Expected arguments '{\"command\": \"df -h\"}', got %s", string(tc.Function.Arguments))
	}
}

func TestStreamAccumulator_Markdown(t *testing.T) {
	ctx := context.Background()
	input := make(chan *ChatResponse)

	// Start transformer
	output := StreamAccumulator(ctx, input, true)

	// Feed chunks
	chunks := []*ChatResponse{
		{
			Message: Message{Content: "Here is some code:\n```go\nfmt.P"},
		},
		{
			Message: Message{Content: "rintln(\"hello\")\n```\nAnd more text."},
		},
	}

	go func() {
		for _, c := range chunks {
			input <- c
		}
		close(input)
	}()

	// Read first response
	acc1 := <-output
	expected1 := "Here is some code:\n```go\nfmt.P\n```"
	if acc1.Content != expected1 {
		t.Errorf("Expected temporarily terminated content %q, got %q", expected1, acc1.Content)
	}

	// Read second response
	acc2 := <-output
	expected2 := "Here is some code:\n```go\nfmt.Println(\"hello\")\n```\nAnd more text."
	if acc2.Content != expected2 {
		t.Errorf("Expected fully terminated content %q, got %q", expected2, acc2.Content)
	}
}

func TestStreamAccumulator_ToolCallSplitChunks_MissingIndexOnLaterChunk(t *testing.T) {
	ctx := context.Background()
	input := make(chan *ChatResponse)
	output := StreamAccumulator(ctx, input, false)

	// First chunk: full tool call metadata including index and id
	idx0 := 0
	chunks := []*ChatResponse{
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Index: &idx0,
						ID:    "call_1",
						Type:  "function",
						Function: FunctionCall{
							Name:      "open_file",
							Arguments: json.RawMessage(""),
						},
					},
				},
			},
		},
		// Second chunk: only provides the function arguments (no index, no id)
		{
			Message: Message{
				ToolCalls: []ToolCall{
					{
						Function: FunctionCall{
							Arguments: json.RawMessage(`{"path":"pkg/webagent/types.go"}`),
						},
					},
				},
			},
		},
	}

	go func() {
		for _, c := range chunks {
			input <- c
		}
		close(input)
	}()

	var lastAcc *AccumulatedResponse
	for acc := range output {
		lastAcc = acc
	}

	if lastAcc == nil {
		t.Fatal("Expected at least one response")
	}

	// Expect a single tool call that merged the arguments into the original call
	if len(lastAcc.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call after merge, got %d", len(lastAcc.ToolCalls))
	}
	tc := lastAcc.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("Expected merged tool call ID 'call_1', got %s", tc.ID)
	}
	if string(tc.Function.Arguments) != `{"path":"pkg/webagent/types.go"}` {
		t.Errorf("Expected merged arguments %q, got %q", `{"path":"pkg/webagent/types.go"}`, string(tc.Function.Arguments))
	}
}
