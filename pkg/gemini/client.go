package gemini

import (
	"context"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"google.golang.org/genai"
)

// GeminiClient handles interaction with the Google Gemini API using google.golang.org/genai
type GeminiClient struct {
	client *genai.Client
}

// NewGeminiClient creates a new Gemini client
func NewGeminiClient(apiKey string) *GeminiClient {
    // A context is required for genai.NewClient
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
        APIKey: apiKey,
    })
    if err != nil {
        // Logically, we should return an error, but the interface dictates
        // returning *GeminiClient. The SDK panic/fails on missing API key only upon use,
        // or we can handle it.
        panic(err)
    }

	return &GeminiClient{
		client: client,
	}
}

// Chat handles a non-streaming request to Gemini and returns the full ChatResponse
func (c *GeminiClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	geminiReq, config, err := ToGeminiRequest(req)
    if err != nil {
        return nil, err
    }

	resp, err := c.client.Models.GenerateContent(ctx, req.Model, geminiReq, config)
	if err != nil {
		return nil, err
	}

	return ToChatResponse(resp), nil
}

// ChatStreamed handles the streaming request to Gemini
func (c *GeminiClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	defer close(ch)

	geminiReq, config, err := ToGeminiRequest(req)
    if err != nil {
        return err
    }

	stream := c.client.Models.GenerateContentStream(ctx, req.Model, geminiReq, config)

    for chunk, err := range stream {
        if err != nil {
            return err
        }
        ch <- ToChatResponse(chunk)
    }

	return nil
}

// Verify interface compliance
var _ ai.ChatClientInterface = (*GeminiClient)(nil)
