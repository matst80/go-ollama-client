package openai

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

const (
	defaultOpenAIIntegrationURL   = "http://127.0.0.1:8081"
	defaultOpenAIIntegrationModel = "current_model.gguf"
)

func TestOpenAIClient_Integration_Chat_LlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	baseURL, model, apiKey := openAIIntegrationConfig()
	if !openAIIntegrationServerReachable(t, baseURL) {
		t.Skipf("skipping integration test because llama.cpp server is not reachable at %s", baseURL)
	}

	client := NewOpenAIClient(baseURL, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := ai.NewChatRequest(model)
	req.AddMessage(ai.MessageRoleUser, "Reply with the exact word READY and nothing else.")

	resp, err := client.Chat(ctx, *req)
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}

	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	if strings.TrimSpace(resp.Message.Content) == "" {
		t.Fatal("expected non-empty response content")
	}

	t.Logf("Model: %s", resp.Model)
	t.Logf("Response: %q", resp.Message.Content)
	if !strings.Contains(strings.ToLower(resp.Message.Content), "ready") {
		t.Fatalf("expected response to contain READY, got %q", resp.Message.Content)
	}
	if resp.DoneReason == "" {
		t.Log("response did not include a done_reason")
	}
}

func TestOpenAIClient_Integration_ChatStreamed_LlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	baseURL, model, apiKey := openAIIntegrationConfig()
	if !openAIIntegrationServerReachable(t, baseURL) {
		t.Skipf("skipping integration test because llama.cpp server is not reachable at %s", baseURL)
	}

	client := NewOpenAIClient(baseURL, apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := ai.NewChatRequest(model)
	req.AddMessage(ai.MessageRoleUser, "Reply with the exact phrase streaming works and nothing else.")

	ch := make(chan *ai.ChatResponse, 128)
	errCh := make(chan error, 1)
	go func() {
		errCh <- client.ChatStreamed(ctx, *req, ch)
	}()

	var content strings.Builder
	seenDone := false
	for resp := range ch {
		if resp == nil {
			continue
		}
		content.WriteString(resp.Message.Content)
		if resp.Done {
			seenDone = true
		}
	}

	if err := <-errCh; err != nil {
		t.Fatalf("ChatStreamed failed: %v", err)
	}

	finalContent := strings.TrimSpace(content.String())
	if finalContent == "" {
		t.Fatal("expected non-empty streamed response content")
	}

	t.Logf("Streamed response: %q", finalContent)
	if !strings.Contains(strings.ToLower(finalContent), "streaming works") {
		t.Fatalf("expected streamed response to contain 'streaming works', got %q", finalContent)
	}
	if !seenDone {
		t.Log("stream completed without an explicit done chunk")
	}
}

func openAIIntegrationConfig() (baseURL string, model string, apiKey string) {
	baseURL = os.Getenv("OPENAI_TEST_URL")
	if baseURL == "" {
		baseURL = defaultOpenAIIntegrationURL
	}

	model = os.Getenv("OPENAI_TEST_MODEL")
	if model == "" {
		model = defaultOpenAIIntegrationModel
	}

	apiKey = os.Getenv("OPENAI_TEST_API_KEY")
	return baseURL, model, apiKey
}

func openAIIntegrationServerReachable(t *testing.T, baseURL string) bool {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/health", strings.TrimRight(baseURL, "/")), nil)
	if err != nil {
		t.Fatalf("failed to build health check request: %v", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}
