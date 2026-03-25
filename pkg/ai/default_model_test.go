package ai

import (
	"context"
	"testing"
)

// mockDefaultChatClient is a simple mock for testing default model logic
type mockDefaultChatClient struct {
	defaultModel string
	lastModel    string
}

func (m *mockDefaultChatClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if req.Model == "" {
		req.Model = m.defaultModel
	}
	m.lastModel = req.Model
	return &ChatResponse{BaseResponse: &BaseResponse{Done: true}}, nil
}

func (m *mockDefaultChatClient) ChatStreamed(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
	if req.Model == "" {
		req.Model = m.defaultModel
	}
	m.lastModel = req.Model
	ch <- &ChatResponse{BaseResponse: &BaseResponse{Done: true}}
	close(ch)
	return nil
}

func (m *mockDefaultChatClient) GetVersion(ctx context.Context) (string, error) {
	return "mock", nil
}

func TestDefaultModel(t *testing.T) {
	ctx := context.Background()
	client1 := &mockDefaultChatClient{defaultModel: "model-1"}

	// Create request with empty model
	req := NewDefaultChatRequest()
	session := NewAgentSession(ctx, client1, req, NewDefaultAgentState())

	// First request
	err := session.SendUserMessage(ctx, "hello")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	// Wait for response to ensure the goroutine finished
	<-session.Recv()

	if client1.lastModel != "model-1" {
		t.Errorf("expected model-1, got %s", client1.lastModel)
	}

	// Change client
	client2 := &mockDefaultChatClient{defaultModel: "model-2"}
	session.SetClient(client2)

	// Second request
	err = session.SendUserMessage(ctx, "hello again")
	if err != nil {
		t.Fatalf("SendUserMessage failed: %v", err)
	}

	// Wait for response
	<-session.Recv()

	if client2.lastModel != "model-2" {
		t.Errorf("expected model-2, got %s", client2.lastModel)
	}
}
