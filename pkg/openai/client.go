package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// OpenAIClient handles interaction with the OpenAI API
type OpenAIClient struct {
	client  *ai.ApiClient
	logPath string
}

type OpenAIEndpoint string

const (
	ChatEndpoint OpenAIEndpoint = "/v1/chat/completions"
)

// NewOpenAIClient creates a new OpenAI client
func NewOpenAIClient(url string, apiKey string) *OpenAIClient {
	return &OpenAIClient{client: ai.NewApiClient(url, map[string]string{"Authorization": fmt.Sprintf("Bearer %s", apiKey)})}
}

// WithLogFile sets the path to the log file where all OpenAI response lines will be stored
func (c *OpenAIClient) WithLogFile(path string) *OpenAIClient {
	c.logPath = path
	return c
}

// Chat handles a non-streaming request to OpenAI and returns the full ChatResponse
func (c *OpenAIClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	// Convert to strongly-typed OpenAI request where function arguments are JSON strings.
	oaReq := ToOpenAIChatRequest(&req)
	oaReq.Stream = false

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), oaReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenAI request failed with status %d", resp.StatusCode)
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

// ChatStreamed handles the streaming request to OpenAI and returns an error if the request or streaming fails.
func (c *OpenAIClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	// Convert to strongly-typed OpenAI request where function arguments are JSON strings.
	oaReq := ToOpenAIChatRequest(&req)
	oaReq.Stream = true
	defer close(ch)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), oaReq)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("OpenAI request failed with status %d", resp.StatusCode)
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
