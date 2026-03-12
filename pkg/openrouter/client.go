package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/matst80/go-ollama-client/pkg/ai"
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
	c.logPath = path
	return c
}

// Chat handles a non-streaming request to OpenRouter and returns the full ChatResponse
func (c *OpenRouterClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenRouter request failed with status %d", resp.StatusCode)
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
	defer close(ch)

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("OpenRouter request failed with status %d", resp.StatusCode)
	}

	var chatResp ChatCompletionChunk

	handler := ai.DataJsonChunkReader(&chatResp, func(chunk *ChatCompletionChunk) bool {
		ch <- chunk.ToChatResponse()
		chatResp = ChatCompletionChunk{} // reset for next chunk
		return false
	})

	// // Use shared ChunkReader to iterate trimmed lines
	// handler2 := func(line []byte) (stop bool) {
	// 	// Log the line if requested (ChunkReader passes a trimmed line)
	// 	if c.logPath != "" {
	// 		if f, err := os.OpenFile(c.logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644); err == nil {
	// 			f.Write(line)
	// 			f.Write([]byte("\n"))
	// 			f.Close()
	// 		}
	// 	}

	// 	// If the line is the done marker, stop processing
	// 	if bytes.Equal(line, DONE) {
	// 		return true
	// 	}

	// 	// Expect lines to start with the data prefix
	// 	if !bytes.HasPrefix(line, DATA_PREFIX) {
	// 		return false
	// 	}

	// 	payload := line[len(DATA_PREFIX):]
	// 	if len(payload) == 0 {
	// 		return false
	// 	}

	// 	var chatResp ChatCompletionChunk
	// 	if err := json.Unmarshal(payload, &chatResp); err != nil {
	// 		log.Printf("error parsing: %s, err: %s", payload, err)
	// 		return false
	// 	}

	// 	ch <- chatResp.ToChatResponse()
	// 	return false
	// }

	if err := ai.ChunkReader(ctx, resp.Body, handler); err != nil {
		return err
	}

	return nil
}
