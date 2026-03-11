package ollama

import (
	"context"
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
			// Simulate a streamed response
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

	session := NewAgentSession(ctx, mockClient, req, WithAccumulator())
	defer session.Stop()

	err := session.SendUserMessage(ctx, "Hi")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	// Verify the user message was added to the request
	if len(req.Messages) == 0 || req.Messages[0].Content != "Hi" {
		t.Errorf("expected user message 'Hi', got %+v", req.Messages)
	}

	// Verify the response is received on the global channel
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
		if !res.Chunk.Done {
			t.Error("expected Done=true")
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for response 2")
	}

	// Wait a bit for the goroutine to append the assistant message to history
	time.Sleep(50 * time.Millisecond)

	// Verify assistant message was added to history
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

	session := NewAgentSession(ctx, mockClient, req, WithAccumulator())
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

	// Verify tool results were added
	if len(req.Messages) == 0 || req.Messages[0].Role != MessageRoleTool {
		t.Errorf("expected tool message, got %+v", req.Messages)
	}

	// Verify response
	select {
	case res := <-session.Recv():
		if res.Content != "Result received" {
			t.Errorf("expected 'Result received', got %q", res.Content)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for response")
	}
}
