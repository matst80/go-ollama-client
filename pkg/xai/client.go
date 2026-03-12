package xai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"net/http"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// XAIClient handles interaction with the xAI API
type XAIClient struct {
	client  *ai.ApiClient
	logPath string
}

type XAIEndpoint string

const (
	ChatEndpoint XAIEndpoint = "chat/completions"
)

// NewXAIClient creates a new xAI client
func NewXAIClient(url string, apiKey string) *XAIClient {
	return &XAIClient{client: ai.NewApiClient(url, map[string]string{"Authorization": fmt.Sprintf("Bearer %s", apiKey)})}
}

// WithLogFile sets the path to the log file where all xAI response lines will be stored
func (c *XAIClient) WithLogFile(path string) *XAIClient {
	c.logPath = path
	return c
}

/* Chat handles a non-streaming request to xAI and returns the full ChatResponse */
func (c *XAIClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	req.Stream = false

	// Convert to strongly-typed XAI request where function arguments are JSON strings.
	xReq := ToXAIChatRequest(&req)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), xReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("xAI request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ai.ChatResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

/* ChatStreamed handles the streaming request to xAI and returns an error if the request or streaming fails. */
func (c *XAIClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	req.Stream = true
	defer close(ch)

	// Convert to strongly-typed XAI request where function arguments are JSON strings.
	xReq := ToXAIChatRequest(&req)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), xReq)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("xAI request failed with status %d and error reading body: %w", resp.StatusCode, err)
		}
		// reqJson, err := json.Marshal(xReq)
		// if err != nil {
		// 	return fmt.Errorf("xAI request failed with status %d and error marshaling request: %w", resp.StatusCode, err)
		// }
		//fmt.Printf("Request:\n%s\n", reqJson)
		return fmt.Errorf("xAI request failed with status %d: %s", resp.StatusCode, string(body))
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

/* prepareRequestData removed. Use ToXAIChatRequest (in pkg/xai/types.go) for conversion to strongly-typed XAI request. */
