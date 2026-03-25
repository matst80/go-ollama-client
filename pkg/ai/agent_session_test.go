package ai

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// MockChatClient implements ChatClientInterface for testing.
type MockChatClient struct {
	ChatFunc         func(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	ChatStreamedFunc func(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error
}

func (m *MockChatClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if m.ChatFunc != nil {
		return m.ChatFunc(ctx, req)
	}
	return nil, nil
}

func (m *MockChatClient) ChatStreamed(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
	if m.ChatStreamedFunc != nil {
		return m.ChatStreamedFunc(ctx, req, ch)
	}
	close(ch)
	return nil
}

func TestAgentSession_SendUserMessage(t *testing.T) {
	ctx := context.Background()
	req := NewChatRequest("test-model")

	mockClient := &MockChatClient{
		ChatStreamedFunc: func(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
			ch <- &ChatResponse{
				BaseResponse: &BaseResponse{Done: false},
				Message:      Message{Role: MessageRoleAssistant, Content: "Hello"},
			}
			ch <- &ChatResponse{
				BaseResponse: &BaseResponse{Done: true},
				Message:      Message{Role: MessageRoleAssistant, Content: " world!"},
			}
			close(ch)
			return nil
		},
	}

	session := NewAgentSession(ctx, mockClient, req, NewDefaultAgentState())
	defer session.Stop()

	err := session.SendUserMessage(ctx, "Hi")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	if len(req.Messages) == 0 || req.Messages[0].Content != "Hi" {
		t.Errorf("expected user message 'Hi', got %+v", req.Messages)
	}

	select {
	case res := <-session.Recv():
		if res.Content != "Hello" {
			t.Errorf("expected 'Hello', got %q", res.Content)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for response 1")
	}

	select {
	case res := <-session.Recv():
		if res.Content != "Hello world!" {
			t.Errorf("expected 'Hello world!', got %q", res.Content)
		}
		if res.Chunk == nil || !res.Chunk.Done {
			t.Error("expected Done=true")
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for response 2")
	}

	time.Sleep(50 * time.Millisecond)

	if len(req.Messages) != 2 {
		t.Errorf("expected 2 messages in history, got %d", len(req.Messages))
	} else if req.Messages[1].Content != "Hello world!" {
		t.Errorf("expected assistant content 'Hello world!', got %q", req.Messages[1].Content)
	}
}

func TestAgentSession_SendToolResults(t *testing.T) {
	ctx := context.Background()
	req := NewChatRequest("test-model")

	mockClient := &MockChatClient{
		ChatStreamedFunc: func(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
			ch <- &ChatResponse{
				BaseResponse: &BaseResponse{Done: true},
				Message:      Message{Role: MessageRoleAssistant, Content: "Result received"},
			}
			close(ch)
			return nil
		},
	}

	session := NewAgentSession(ctx, mockClient, req, NewDefaultAgentState())
	defer session.Stop()

	toolMsg := Message{
		Role:       MessageRoleTool,
		Content:    "disk space 50GB",
		ToolCallID: "call_123",
	}

	err := session.SendMessages(ctx, toolMsg)
	if err != nil {
		t.Fatalf("SendMessages failed: %v", err)
	}

	if len(req.Messages) == 0 || req.Messages[0].Role != MessageRoleTool {
		t.Errorf("expected tool message, got %+v", req.Messages)
	}

	select {
	case res := <-session.Recv():
		if res.Content != "Result received" {
			t.Errorf("expected 'Result received', got %q", res.Content)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for response")
	}
}

func TestAgentSession_WithAutoToolExecutor_SendsToolResults(t *testing.T) {
	ctx := context.Background()
	req := NewChatRequest("test-model")

	type callbackResult struct {
		CallID  string
		Content string
		Err     error
	}

	callCount := 0
	mockClient := &MockChatClient{
		ChatStreamedFunc: func(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
			callCount++

			switch callCount {
			case 1:
				ch <- &ChatResponse{
					BaseResponse: &BaseResponse{Done: true},
					Message: Message{
						Role:    MessageRoleAssistant,
						Content: "Need tool",
						ToolCalls: []ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: FunctionCall{
									Name:      "echo",
									Arguments: json.RawMessage(`{"message":"hello from tool"}`),
								},
							},
						},
					},
				}
			case 2:
				if len(req.Messages) < 2 {
					t.Fatalf("expected tool result message in history, got %d messages", len(req.Messages))
				}
				last := req.Messages[len(req.Messages)-1]
				if last.Role != MessageRoleTool {
					t.Fatalf("expected last message role to be tool, got %s", last.Role)
				}
				if last.ToolCallID != "call_1" {
					t.Fatalf("expected tool call id call_1, got %s", last.ToolCallID)
				}
				if last.Content != "tool says hello from tool" {
					t.Fatalf("expected tool content to be delivered, got %q", last.Content)
				}

				ch <- &ChatResponse{
					BaseResponse: &BaseResponse{Done: true},
					Message: Message{
						Role:    MessageRoleAssistant,
						Content: "Final answer",
					},
				}
			default:
				t.Fatalf("unexpected ChatStreamed call %d", callCount)
			}

			close(ch)
			return nil
		},
	}

	var callbackResults []callbackResult
	session := NewAgentSession(ctx, mockClient, req, NewDefaultAgentState(),
		WithAutoToolExecutor(func(ctx context.Context, calls []ToolCall) ([]Message, []AutoToolResult, error) {
			if len(calls) != 1 {
				t.Fatalf("expected exactly one tool call, got %d", len(calls))
			}
			if calls[0].Function.Name != "echo" {
				t.Fatalf("expected tool name echo, got %s", calls[0].Function.Name)
			}

			msg := Message{
				Role:       MessageRoleTool,
				ToolCallID: calls[0].ID,
				Content:    "tool says hello from tool",
			}
			autoResult := AutoToolResult{
				CallID:  calls[0].ID,
				Content: "tool says hello from tool",
				Err:     nil,
			}
			callbackResults = append(callbackResults, callbackResult(autoResult))

			return []Message{msg}, []AutoToolResult{autoResult}, nil
		}),
	)
	defer session.Stop()

	err := session.SendUserMessage(ctx, "Hi")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	var got []AccumulatedResponse
	timeout := time.After(2 * time.Second)
	for len(got) < 2 {
		select {
		case res := <-session.Recv():
			got = append(got, res)
		case <-timeout:
			t.Fatalf("timed out waiting for responses, got %d", len(got))
		}
	}

	if got[0].Content != "Need tool" {
		t.Fatalf("expected first response content %q, got %q", "Need tool", got[0].Content)
	}
	if got[0].Chunk == nil || !got[0].Chunk.Done {
		t.Fatal("expected first response to be done")
	}
	if len(got[0].ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(got[0].ToolCalls))
	}

	if got[1].Content != "Final answer" {
		t.Fatalf("expected second response content %q, got %q", "Final answer", got[1].Content)
	}
	if got[1].Chunk == nil || !got[1].Chunk.Done {
		t.Fatal("expected second response to be done")
	}

	if callCount != 2 {
		t.Fatalf("expected ChatStreamed to be called twice, got %d", callCount)
	}

	if len(callbackResults) != 1 {
		t.Fatalf("expected callback to be invoked once, got %d", len(callbackResults))
	}
	if callbackResults[0].CallID != "call_1" {
		t.Fatalf("expected callback call id call_1, got %s", callbackResults[0].CallID)
	}
	if callbackResults[0].Content != "tool says hello from tool" {
		t.Fatalf("expected callback content %q, got %q", "tool says hello from tool", callbackResults[0].Content)
	}
	if callbackResults[0].Err != nil {
		t.Fatalf("expected callback error to be nil, got %v", callbackResults[0].Err)
	}
}

func TestAgentSession_WithAutoToolExecutor_WithoutCallback(t *testing.T) {
	ctx := context.Background()
	req := NewChatRequest("test-model")

	callCount := 0
	mockClient := &MockChatClient{
		ChatStreamedFunc: func(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
			callCount++

			switch callCount {
			case 1:
				ch <- &ChatResponse{
					BaseResponse: &BaseResponse{Done: true},
					Message: Message{
						Role:    MessageRoleAssistant,
						Content: "Need tool",
						ToolCalls: []ToolCall{
							{
								ID:   "call_2",
								Type: "function",
								Function: FunctionCall{
									Name:      "echo",
									Arguments: json.RawMessage(`{"message":"no callback"}`),
								},
							},
						},
					},
				}
			case 2:
				ch <- &ChatResponse{
					BaseResponse: &BaseResponse{Done: true},
					Message: Message{
						Role:    MessageRoleAssistant,
						Content: "Done",
					},
				}
			default:
				t.Fatalf("unexpected ChatStreamed call %d", callCount)
			}

			close(ch)
			return nil
		},
	}

	session := NewAgentSession(ctx, mockClient, req, NewDefaultAgentState(), WithAutoToolExecutor(func(ctx context.Context, calls []ToolCall) ([]Message, []AutoToolResult, error) {
		if len(calls) != 1 {
			t.Fatalf("expected exactly one tool call, got %d", len(calls))
		}

		msg := Message{
			Role:       MessageRoleTool,
			ToolCallID: calls[0].ID,
			Content:    "tool says no callback",
		}
		autoResult := AutoToolResult{
			CallID:  calls[0].ID,
			Content: "tool says no callback",
			Err:     nil,
		}

		return []Message{msg}, []AutoToolResult{autoResult}, nil
	}))
	defer session.Stop()

	err := session.SendUserMessage(ctx, "Hi")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	timeout := time.After(2 * time.Second)

	select {
	case res := <-session.Recv():
		if res.Content != "Need tool" {
			t.Fatalf("expected first response content %q, got %q", "Need tool", res.Content)
		}
	case <-timeout:
		t.Fatal("timed out waiting for first response")
	}

	select {
	case res := <-session.Recv():
		if res.Content != "Done" {
			t.Fatalf("expected second response content %q, got %q", "Done", res.Content)
		}
	case <-timeout:
		t.Fatal("timed out waiting for second response")
	}

	if callCount != 2 {
		t.Fatalf("expected ChatStreamed to be called twice, got %d", callCount)
	}
}
