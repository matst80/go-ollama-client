package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// OllamaClient handles interaction with the Ollama API
type OllamaClient struct {
	auth string
	URL  string
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
	return &OllamaClient{URL: url}
}

func (c *OllamaClient) WithAuth(auth string) *OllamaClient {
	c.auth = auth
	return c
}

func (c *OllamaClient) post(ctx context.Context, endpoint OllamaEndpoint, data any) (*http.Response, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/%s", c.URL, endpoint), bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *OllamaClient) get(ctx context.Context, endpoint OllamaEndpoint) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/%s", c.URL, endpoint), nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

// Chat handles a non-streaming request to Ollama and returns the full ChatResponse
func (c *OllamaClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	resp, err := c.post(ctx, ChatEndpoint, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp ChatResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

// ChatStreamed handles the streaming request to Ollama and returns an error if the request or streaming fails.
func (c *OllamaClient) ChatStreamed(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {
	defer close(ch)

	resp, err := c.post(ctx, ChatEndpoint, req)
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

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}

		cleanLine := bytes.TrimSpace(line)
		if len(cleanLine) == 0 {
			continue
		}

		var chatResp ChatResponse
		if err := json.Unmarshal(cleanLine, &chatResp); err != nil {
			continue
		}

		if chatResp.Error != nil {
			return fmt.Errorf("ollama error: %s", *chatResp.Error)
		}

		ch <- &chatResp

		if chatResp.Done {
			break
		}
	}
	return nil
}

func (c *OllamaClient) Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	req.Stream = false

	resp, err := c.post(ctx, GenerateEndpoint, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp GenerateResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *OllamaClient) GenerateEmbeddings(ctx context.Context, req EmbeddingsRequest) (*EmbeddingsResponse, error) {
	req.Stream = false

	resp, err := c.post(ctx, GenerateEmbeddingsEndpoint, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var chatResp EmbeddingsResponse
	if err := decoder.Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &chatResp, nil
}

func (c *OllamaClient) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	resp, err := c.get(ctx, ListModelsEndpoint)
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
	resp, err := c.get(ctx, ListRunningModelsEndpoint)
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
	resp, err := c.post(ctx, ShowEndpoint, req)
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

	resp, err := c.post(ctx, CreateEndpoint, req)
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

	resp, err := c.post(ctx, CreateEndpoint, req)
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

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}

		cleanLine := bytes.TrimSpace(line)
		if len(cleanLine) == 0 {
			continue
		}

		var createResp CreateResponse
		if err := json.Unmarshal(cleanLine, &createResp); err != nil {
			continue
		}

		ch <- &createResp

		if createResp.Status == "success" {
			break
		}
	}
	return nil
}

func (c *OllamaClient) CopyModel(ctx context.Context, source, destination string) error {
	_, err := c.post(ctx, CopyModelEndpoint, CopyRequest{
		Source:      source,
		Destination: destination,
	})
	return err
}

func (c *OllamaClient) PullModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error) {
	req.Stream = false

	resp, err := c.post(ctx, PullModelEndpoint, req)
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

	resp, err := c.post(ctx, PullModelEndpoint, req)
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

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}

		cleanLine := bytes.TrimSpace(line)
		if len(cleanLine) == 0 {
			continue
		}

		var pullResp StatusResponse
		if err := json.Unmarshal(cleanLine, &pullResp); err != nil {
			continue
		}

		if pullResp.Error != "" {
			return fmt.Errorf("ollama error: %s", pullResp.Error)
		}

		ch <- &pullResp

		if pullResp.Status == "success" {
			break
		}
	}
	return nil
}

func (c *OllamaClient) PushModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error) {
	req.Stream = false

	resp, err := c.post(ctx, PushModelEndpoint, req)
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

	resp, err := c.post(ctx, PushModelEndpoint, req)
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

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}

		cleanLine := bytes.TrimSpace(line)
		if len(cleanLine) == 0 {
			continue
		}

		var statusResp StatusResponse
		if err := json.Unmarshal(cleanLine, &statusResp); err != nil {
			continue
		}

		if statusResp.Error != "" {
			return fmt.Errorf("ollama error: %s", statusResp.Error)
		}

		ch <- &statusResp

		if statusResp.Status == "success" {
			break
		}
	}
	return nil
}

func (c *OllamaClient) DeleteModel(ctx context.Context, model string) error {
	resp, err := c.post(ctx, DeleteModelEndpoint, DeleteRequest{Model: model})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama request failed with status %d", resp.StatusCode)
	}

	return nil
}

func (c *OllamaClient) GetVersion(ctx context.Context) (string, error) {
	resp, err := c.get(ctx, VersionEndpoint)
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
