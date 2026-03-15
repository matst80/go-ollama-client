package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// OpenRouterClient handles interaction with the OpenRouter API
type OpenRouterClient struct {
	client  *ai.ApiClient
	logPath string
}

type OpenRouterEndpoint string

const (
	ChatEndpoint OpenRouterEndpoint = "api/v1/chat/completions"
)

// NewOpenRouterClient creates a new OpenRouter client
func NewOpenRouterClient(url string, apiKey string) *OpenRouterClient {
	return &OpenRouterClient{client: ai.NewApiClient(url, map[string]string{"Authorization": fmt.Sprintf("Bearer %s", apiKey)})}
}

// WithLogFile sets the path to the log file where all OpenRouter response lines will be stored
func (c *OpenRouterClient) WithLogFile(path string) *OpenRouterClient {
	// Forward to the underlying ApiClient so logging is handled in one place.
	// Keep the local `logPath` field for backward compatibility.
	c.client.WithLogFile(path)
	c.logPath = path
	return c
}

// Chat handles a non-streaming request to OpenRouter and returns the full ChatResponse
func (c *OpenRouterClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	req.Stream = false

	// Convert to strongly-typed OpenRouter request where function arguments are JSON strings.
	orReq := ToOpenRouterChatRequest(&req)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), orReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		reqBody, _ := json.Marshal(orReq)
		log.Printf("OpenRouter error: status=%d", resp.StatusCode)
		log.Printf("Request\n%s", reqBody)
		log.Printf("Response\n%s", body)
		return nil, fmt.Errorf("OpenRouter request failed with status %d: %s", resp.StatusCode, string(body))
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ai.ChatResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

var DATA_PREFIX = []byte("data: ")
var DONE = []byte("[DONE]")

// ChatStreamed handles the streaming request to OpenRouter and returns an error if the request or streaming fails.
func (c *OpenRouterClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	req.Stream = true
	defer close(ch)

	// Convert to strongly-typed OpenRouter request where function arguments are JSON strings.
	orReq := ToOpenRouterChatRequest(&req)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), orReq)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		reqBody, _ := json.Marshal(orReq)
		log.Printf("OpenRouter error: status=%d, request=%s, response=%s", resp.StatusCode, string(reqBody), string(body))
		return fmt.Errorf("OpenRouter request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatCompletionChunk

	handler := ai.DataJsonChunkReader(&chatResp, func(chunk *ChatCompletionChunk) bool {
		ch <- chunk.ToChatResponse()
		chatResp = ChatCompletionChunk{} // reset for next chunk
		return false
	})

	if err := ai.ChunkReader(ctx, resp.Body, handler); err != nil {
		return err
	}

	return nil
}
