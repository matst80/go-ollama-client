package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

func TestClientDefaultOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("expected /api/chat, got %s", r.URL.Path)
		}
		var req OllamaChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if req.Options == nil {
			t.Fatal("expected options to be set in request")
		}
		if req.Options.ContextWindowSize != 8192 {
			t.Errorf("expected context_window_size 8192, got %d", req.Options.ContextWindowSize)
		}
		if req.Options.Temperature != 0.7 {
			t.Errorf("expected temperature 0.7, got %f", req.Options.Temperature)
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{Done: true},
			Message: ai.Message{Role: ai.MessageRoleAssistant, Content: "Hello"},
		})
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	client.WithOptions(&ModelOptions{
		ContextWindowSize: 8192,
		Temperature:       0.7,
	})

	req := ai.NewChatRequest("test-model").AddMessage(ai.MessageRoleUser, "Hi")
	_, err := client.Chat(context.Background(), *req)
	if err != nil {
		t.Fatal(err)
	}
}

func TestClientDefaultOptionsStreamed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req OllamaChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		if req.Options == nil {
			t.Fatal("expected options to be set in request")
		}
		if req.Options.ContextWindowSize != 4096 {
			t.Errorf("expected context_window_size 4096, got %d", req.Options.ContextWindowSize)
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{Done: true},
			Message: ai.Message{Role: ai.MessageRoleAssistant, Content: "Hello"},
		})
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	client.WithOptions(&ModelOptions{
		ContextWindowSize: 4096,
	})

	ch := make(chan *ai.ChatResponse)
	req := ai.NewChatRequest("test-model").AddMessage(ai.MessageRoleUser, "Hi")
	
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := client.ChatStreamed(ctx, *req, ch)
	if err != nil {
		t.Fatal(err)
	}
	for range ch {
	}
}
