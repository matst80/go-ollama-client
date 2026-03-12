package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/matst80/go-ollama-client/pkg/ai"
)

// OllamaClient handles interaction with the Ollama API
type OllamaClient struct {
	client *ai.ApiClient
}

type OllamaEndpoint string

const (
	ChatEndpoint               OllamaEndpoint = "api/chat"
	GenerateEndpoint           OllamaEndpoint = "api/generate"
	GenerateEmbeddingsEndpoint OllamaEndpoint = "api/embed"
	ListModelsEndpoint         OllamaEndpoint = "api/tags"
	ListRunningModelsEndpoint  OllamaEndpoint = "api/ps"
	CreateEndpoint             OllamaEndpoint = "api/create"
	ShowEndpoint               OllamaEndpoint = "api/show"
	CopyModelEndpoint          OllamaEndpoint = "api/copy"
	PullModelEndpoint          OllamaEndpoint = "api/pull"
	PushModelEndpoint          OllamaEndpoint = "api/push"
	DeleteModelEndpoint        OllamaEndpoint = "api/delete"
	VersionEndpoint            OllamaEndpoint = "api/version"
)

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(url string) *OllamaClient {
	return &OllamaClient{client: ai.NewApiClient(url, map[string]string{})}
}

func (c *OllamaClient) WithAuth(auth string) *OllamaClient {
	c.client.SetHeaders(map[string]string{
		"Authorization": fmt.Sprintf("Bearer %s", auth),
	})
	return c
}

// Chat handles a non-streaming request to Ollama and returns the full ChatResponse
func (c *OllamaClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(ChatEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ai.ChatResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

// ChatStreamed handles the streaming request to Ollama and returns an error if the request or streaming fails.
func (c *OllamaClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
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
		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}
	chatResp := &ai.ChatResponse{}
	handler := ai.JsonChunkReader(chatResp, func(data *ai.ChatResponse) (stop bool) {
		ch <- data
		done := data.Done
		chatResp = &ai.ChatResponse{} // reset for next chunk
		return done
	})

	err = ai.ChunkReader(ctx, resp.Body, handler)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}

	return nil
}

func (c *OllamaClient) Generate(ctx context.Context, req ai.GenerateRequest) (*ai.GenerateResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(GenerateEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ai.GenerateResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *OllamaClient) GenerateEmbeddings(ctx context.Context, req ai.EmbeddingsRequest) (*ai.EmbeddingsResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(GenerateEmbeddingsEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ai.EmbeddingsResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *OllamaClient) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	resp, err := c.client.GetJson(ctx, string(ListModelsEndpoint))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ListModelsResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *OllamaClient) ListRunningModels(ctx context.Context) (*ListRunningModelsResponse, error) {
	resp, err := c.client.GetJson(ctx, string(ListRunningModelsEndpoint))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var res ListRunningModelsResponse
	if err := decoder.Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &res, nil
}

func (c *OllamaClient) ShowModelDetails(ctx context.Context, model string) (*ModelDetailsResponse, error) {
	req := ModelDetailsRequest{Model: model}
	resp, err := c.client.PostJson(ctx, string(ShowEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var res ModelDetailsResponse
	if err := decoder.Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &res, nil
}

func (c *OllamaClient) CreateModel(ctx context.Context, req CreateRequest) (*CreateResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(CreateEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var createResp CreateResponse
	if err := decoder.Decode(&createResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &createResp, nil
}

func (c *OllamaClient) CreateModelStreamed(ctx context.Context, req CreateRequest, ch chan *CreateResponse) error {
	defer close(ch)

	resp, err := c.client.PostJson(ctx, string(CreateEndpoint), req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}
	var createResp CreateResponse
	jsonHandler := ai.JsonChunkReader(&createResp, func(data *CreateResponse) (stop bool) {
		ch <- data
		createResp = CreateResponse{}   // reset for next chunk
		return data.Status == "success" // stop if creation is complete
	})

	err = ai.ChunkReader(ctx, resp.Body, jsonHandler)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	return nil
}

func (c *OllamaClient) CopyModel(ctx context.Context, source, destination string) error {
	_, err := c.client.PostJson(ctx, string(CopyModelEndpoint), CopyRequest{
		Source:      source,
		Destination: destination,
	})
	return err
}

func (c *OllamaClient) PullModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(PullModelEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var pullResp StatusResponse
	if err := decoder.Decode(&pullResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &pullResp, nil
}

func (c *OllamaClient) PullModelStreamed(ctx context.Context, req PushPullRequest, ch chan *StatusResponse) error {
	defer close(ch)
	req.Stream = true

	resp, err := c.client.PostJson(ctx, string(PullModelEndpoint), req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	err = ai.ChunkReader(ctx, resp.Body, func(line []byte) (stop bool) {
		var pullResp StatusResponse
		if err := json.Unmarshal(line, &pullResp); err != nil {
			// skip malformed chunk
			return false
		}

		ch <- &pullResp

		if pullResp.Status == "success" {
			return true
		}
		return false
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	return nil
}

func (c *OllamaClient) PushModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error) {
	req.Stream = false

	resp, err := c.client.PostJson(ctx, string(PushModelEndpoint), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var statusResp StatusResponse
	if err := decoder.Decode(&statusResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &statusResp, nil
}

func (c *OllamaClient) PushModelStreamed(ctx context.Context, req PushPullRequest, ch chan *StatusResponse) error {
	defer close(ch)
	req.Stream = true

	resp, err := c.client.PostJson(ctx, string(PushModelEndpoint), req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	err = ai.ChunkReader(ctx, resp.Body, func(line []byte) (stop bool) {
		var statusResp StatusResponse
		if err := json.Unmarshal(line, &statusResp); err != nil {
			// skip malformed chunk
			return false
		}

		ch <- &statusResp
		return false
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	return nil
}

// func (c *OllamaClient) DeleteModel(ctx context.Context, model string) error {
// 	resp, err := c.client.Delete(ctx, string(DeleteModelEndpoint), DeleteRequest{Model: model})
// 	if err != nil {
// 		return err
// 	}
// 	defer resp.Body.Close()

// 	if resp.StatusCode != http.StatusOK {
// 		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
// 	}

// 	return nil
// }

func (c *OllamaClient) GetVersion(ctx context.Context) (string, error) {
	resp, err := c.client.GetJson(ctx, string(VersionEndpoint))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var res VersionResponse
	if err := decoder.Decode(&res); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return res.Version, nil
}
