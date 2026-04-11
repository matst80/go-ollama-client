package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

func TestOpenAIClient_ChatStreamed_EmitsDoneOnSSEDoneMarker(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != string(ChatEndpoint) {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			`data: {"id":"chunk-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"streaming "}}]}`,
			`data: {"id":"chunk-2","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"content":"works"}}]}`,
			`data: [DONE]`,
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected response writer to support flushing")
		}

		for _, chunk := range chunks {
			if _, err := fmt.Fprintf(w, "%s\n\n", chunk); err != nil {
				t.Fatalf("failed to write chunk: %v", err)
			}
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := NewOpenAIClient(server.URL, "")
	req := ai.NewChatRequest("test-model")
	req.AddMessage(ai.MessageRoleUser, "test")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch := make(chan *ai.ChatResponse, 8)
	errCh := make(chan error, 1)
	go func() {
		errCh <- client.ChatStreamed(ctx, *req, ch)
	}()

	var responses []*ai.ChatResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	if err := <-errCh; err != nil {
		t.Fatalf("ChatStreamed failed: %v", err)
	}

	if len(responses) != 3 {
		t.Fatalf("expected 3 streamed responses, got %d", len(responses))
	}

	if responses[0].Message.Content != "streaming " {
		t.Fatalf("expected first chunk content %q, got %q", "streaming ", responses[0].Message.Content)
	}

	if responses[1].Message.Content != "works" {
		t.Fatalf("expected second chunk content %q, got %q", "works", responses[1].Message.Content)
	}

	if !responses[2].Done {
		t.Fatal("expected final streamed response to have Done=true")
	}

	if responses[2].Message.Content != "" {
		t.Fatalf("expected final done chunk to have empty content, got %q", responses[2].Message.Content)
	}

	if responses[2].Model != "test-model" {
		t.Fatalf("expected final done chunk model %q, got %q", "test-model", responses[2].Model)
	}
}
