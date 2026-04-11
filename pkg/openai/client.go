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
	// Forward to the underlying ApiClient so logging is handled in one place.
	// Keep the local `logPath` field for backward compatibility.
	if c.client != nil {
		c.client.WithLogFile(path)
	}
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
	var chatCompletion ChatCompletion
	if err := decoder.Decode(&chatCompletion); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return chatCompletion.ToChatResponse(), nil
}

var DATA_PREFIX = []byte("data: ")
var DONE = []byte("[DONE]")

// ChatStreamed handles the streaming request to OpenAI and returns an error if the request or streaming fails.
func (c *OpenAIClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	// Convert to strongly-typed OpenAI request where function arguments are JSON strings.
	oaReq := ToOpenAIChatRequest(&req)
	oaReq.Stream = true
	defer close(ch)
	sawDone := false

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

	handler := ai.DataJsonChunkReader(func(chunk *ChatCompletionChunk) bool {
		response := chunk.ToChatResponse()
		if response.Done {
			sawDone = true
		}
		ch <- response
		return false
	})

	if err := ai.ChunkReader(ctx, resp.Body, handler); err != nil {
		return err
	}

	if !sawDone {
		ch <- &ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{
				Model: oaReq.Model,
				Done:  true,
			},
			Message: ai.Message{Role: ai.MessageRoleAssistant},
		}
	}

	return nil
}
