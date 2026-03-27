package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"google.golang.org/genai"
)

// GeminiClient handles interaction with the Google Gemini API using google.golang.org/genai
type GeminiClient struct {
	client       *genai.Client
	defaultModel string
	logPath      string
}

// NewGeminiClient creates a new Gemini client
func NewGeminiClient(apiKey string) *GeminiClient {
	// A context is required for genai.NewClient
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
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

// WithLogFile sets the path to the log file
func (c *GeminiClient) WithLogFile(path string) *GeminiClient {
	c.logPath = path
	return c
}

// WithDefaultModel sets the default model
func (c *GeminiClient) WithDefaultModel(model string) *GeminiClient {
	c.defaultModel = model
	return c
}

// Chat handles a non-streaming request to Gemini and returns the full ChatResponse
func (c *GeminiClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	if req.Model == "" {
		req.Model = c.defaultModel
	}

	geminiReq, config, err := ToGeminiRequest(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.client.Models.GenerateContent(ctx, req.Model, geminiReq, config)
	if err != nil {
		if c.logPath != "" {
			reqJSON, _ := json.Marshal(geminiReq)
			configJSON, _ := json.Marshal(config)
			log.Printf("Gemini error: %v", err)
			log.Printf("Request Model: %s", req.Model)
			log.Printf("Request Content: %s", string(reqJSON))
			log.Printf("Request Config: %s", string(configJSON))
		}
		return nil, fmt.Errorf("Gemini request failed: %w", err)
	}

	return ToChatResponse(resp), nil
}

// ChatStreamed handles the streaming request to Gemini
func (c *GeminiClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	defer close(ch)

	if req.Model == "" {
		req.Model = c.defaultModel
	}

	geminiReq, config, err := ToGeminiRequest(req)
	if err != nil {
		return err
	}

	stream := c.client.Models.GenerateContentStream(ctx, req.Model, geminiReq, config)

	for chunk, err := range stream {
		if err != nil {
			if c.logPath != "" {
				log.Printf("Gemini stream error: %v", err)
			}
			return fmt.Errorf("Gemini stream error: %w", err)
		}
		ch <- ToChatResponse(chunk)
	}

	return nil
}

// Verify interface compliance
var _ ai.ChatClientInterface = (*GeminiClient)(nil)
