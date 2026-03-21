package github

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

func TestGitHubClient_ChatStreamed_ToolCalls(t *testing.T) {
	// Mock server that returns a stream of tool call chunks
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Sending chunks for a tool call: "get_weather" with arguments "{\"city\": \"Berlin\"}"
		chunks := []string{
			`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}`,
			`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\": "}}]}}]}`,
			`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Berl"}}]}}]}`,
			`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"in\"}"}}]}}]}`,
			`data: [DONE]`,
		}

		for _, chunk := range chunks {
			_, _ = w.Write([]byte(chunk + "\n"))
		}
	}))
	defer server.Close()

	client := NewGitHubClient("test-token", "")
	client.client.BaseUrl = server.URL
	req := ai.NewChatRequest("test-model")
	ch := make(chan *ai.ChatResponse, 10)

	ctx := context.Background()
	err := client.ChatStreamed(ctx, *req, ch)
	if err != nil {
		t.Fatalf("ChatStreamed failed: %v", err)
	}

	var allResponses []*ai.ChatResponse
	for resp := range ch {
		allResponses = append(allResponses, resp)
	}

	if len(allResponses) == 0 {
		t.Fatal("expected responses, got none")
	}

	// Verify the tool call accumulation if we were to use StreamAccumulator
	accCh := ai.StreamAccumulator(ctx, (chan *ai.ChatResponse)(nil), false)
	// Actually, we need to feed the responses into a channel for StreamAccumulator
	inputCh := make(chan *ai.ChatResponse, len(allResponses))
	for _, r := range allResponses {
		inputCh <- r
	}
	close(inputCh)

	accCh = ai.StreamAccumulator(ctx, inputCh, false)
	var lastAcc *ai.AccumulatedResponse
	for acc := range accCh {
		lastAcc = acc
	}

	if lastAcc == nil {
		t.Fatal("expected accumulated response")
	}

	if len(lastAcc.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(lastAcc.ToolCalls))
	}

	tc := lastAcc.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", tc.Function.Name)
	}

	expectedArgs := `{"city": "Berlin"}`
	if string(tc.Function.Arguments) != expectedArgs {
		t.Errorf("expected arguments %s, got %s", expectedArgs, string(tc.Function.Arguments))
	}
}

func TestGitHubClient_Chat_UsesCopilotEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected path /chat/completions, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected Authorization header Bearer test-token, got %s", got)
		}
		if got := r.Header.Get("X-GitHub-Api-Version"); got != "" {
			t.Errorf("expected no X-GitHub-Api-Version header, got %s", got)
		}
		if got := r.Header.Get("Accept"); got != "" {
			t.Errorf("expected no Accept header, got %s", got)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "chatcmpl-1",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "hello from copilot",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	client := NewGitHubClient("test-token", "")
	client.client.BaseUrl = server.URL

	req := ai.NewChatRequest("test-model")
	req.Messages = []ai.Message{
		{Role: ai.MessageRoleUser, Content: "hi"},
	}

	resp, err := client.Chat(context.Background(), *req)
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Message.Content != "hello from copilot" {
		t.Errorf("expected assistant content hello from copilot, got %s", resp.Message.Content)
	}
}
