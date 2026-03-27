package github

import (
	"context"
	"fmt"
	"sync"

	"github.com/github/copilot-sdk/go"
	"github.com/matst80/go-ai-agent/pkg/ai"
)

// GitHubClient handles interaction with the GitHub Copilot CLI via copilot-sdk
type GitHubClient struct {
	client       *copilot.Client
	defaultModel string

	mu       sync.RWMutex
	sessions map[string]*copilot.Session
}

// NewGitHubClient creates a new GitHub client backed by Copilot SDK
func NewGitHubClient() *GitHubClient {
	c := copilot.NewClient(nil)
	if err := c.Start(context.Background()); err != nil {
		fmt.Printf("Warning: failed to start copilot client: %v\n", err)
	}
	return &GitHubClient{
		client:   c,
		sessions: make(map[string]*copilot.Session),
	}
}

// WithLogFile is retained for interface compatibility but ignored as we use copilot-sdk.
func (c *GitHubClient) WithLogFile(path string) *GitHubClient {
	return c
}

// WithDefaultModel sets the default model to use if no model is specified in a request.
func (c *GitHubClient) WithDefaultModel(model string) *GitHubClient {
	c.defaultModel = model
	return c
}

// getOrCreateSession retrieves an existing Copilot session or creates a new one.
func (c *GitHubClient) getOrCreateSession(ctx context.Context, sessionID string, model string) (*copilot.Session, error) {
	if model == "" {
		model = c.defaultModel
	}

	if sessionID != "" {
		c.mu.RLock()
		sess, ok := c.sessions[sessionID]
		c.mu.RUnlock()
		if ok {
			return sess, nil
		}

		c.mu.Lock()
		defer c.mu.Unlock()
		// Double-check locking
		if sess, ok := c.sessions[sessionID]; ok {
			return sess, nil
		}
	}

	sess, err := c.client.CreateSession(ctx, &copilot.SessionConfig{
		Model:               model,
		OnPermissionRequest: copilot.PermissionHandler.ApproveAll,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create copilot session: %w", err)
	}

	if sessionID != "" {
		c.sessions[sessionID] = sess
	}

	return sess, nil
}

// Chat handles a non-streaming request to GitHub Models
func (c *GitHubClient) Chat(ctx context.Context, req ai.ChatRequest) (*ai.ChatResponse, error) {
	sess, err := c.getOrCreateSession(ctx, req.SessionID, req.Model)
	if err != nil {
		return nil, err
	}

	if req.Model != "" && req.Model != c.defaultModel {
		if err := sess.SetModel(ctx, req.Model); err != nil {
			return nil, fmt.Errorf("failed to set model: %w", err)
		}
	}

	var latestMessage string
	if len(req.Messages) > 0 {
		latestMessage = req.Messages[len(req.Messages)-1].Content
	}

	event, err := sess.SendAndWait(ctx, copilot.MessageOptions{
		Prompt: latestMessage,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to send message: %w", err)
	}

	var content string
	if event != nil && event.Data.Content != nil {
		content = *event.Data.Content
	}

	return &ai.ChatResponse{
		BaseResponse: &ai.BaseResponse{
			Done: true,
		},
		Message: ai.Message{
			Role:    ai.MessageRoleAssistant,
			Content: content,
		},
	}, nil
}

// ChatStreamed handles the streaming request to GitHub Models
func (c *GitHubClient) ChatStreamed(ctx context.Context, req ai.ChatRequest, ch chan *ai.ChatResponse) error {
	defer close(ch)

	sess, err := c.getOrCreateSession(ctx, req.SessionID, req.Model)
	if err != nil {
		return err
	}

	if req.Model != "" && req.Model != c.defaultModel {
		if err := sess.SetModel(ctx, req.Model); err != nil {
			return fmt.Errorf("failed to set model: %w", err)
		}
	}

	var latestMessage string
	if len(req.Messages) > 0 {
		latestMessage = req.Messages[len(req.Messages)-1].Content
	}

	done := make(chan error, 1) // Buffer of 1 prevents deadlock if both error and done happen

	msgID, err := sess.Send(ctx, copilot.MessageOptions{
		Prompt: latestMessage,
	})
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}

	// Make sure we stop sending to ch after we unsubscribe and exit
	chClosed := false
	var chMu sync.Mutex

	// Register event handler AFTER sending so msgID is known and we don't have a race condition
	unsubscribe := sess.On(func(event copilot.SessionEvent) {
		// Filter events: only process events for our interaction.
		if event.Data.InteractionID != nil && *event.Data.InteractionID != msgID {
			return
		}

		chMu.Lock()
		defer chMu.Unlock()
		if chClosed {
			return
		}

		switch event.Type {
		case copilot.SessionEventType("assistant.message.chunk"):
			if event.Data.DeltaContent != nil {
				ch <- &ai.ChatResponse{
					BaseResponse: &ai.BaseResponse{
						Done: false,
					},
					Message: ai.Message{
						Role:    ai.MessageRoleAssistant,
						Content: *event.Data.DeltaContent,
					},
				}
			}
		case copilot.SessionEventType("assistant.error"):
			var errMsg string
			if event.Data.ErrorReason != nil {
				errMsg = *event.Data.ErrorReason
			} else if event.Data.ErrorType != nil {
				errMsg = *event.Data.ErrorType
			} else {
				errMsg = "unknown assistant error"
			}
			select {
			case done <- fmt.Errorf("assistant error: %s", errMsg):
			default:
			}
		case copilot.SessionEventType("assistant.message.done"):
			select {
			case done <- nil:
			default:
			}
		}
	})

	defer func() {
		unsubscribe()
		chMu.Lock()
		chClosed = true
		chMu.Unlock()
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		if err == nil {
			chMu.Lock()
			if !chClosed {
				ch <- &ai.ChatResponse{
					BaseResponse: &ai.BaseResponse{
						Done: true,
					},
				}
			}
			chMu.Unlock()
		}
		return err
	}
}

// ModelInfo represents information about a model from the GitHub catalog
type ModelInfo struct {
	ID        string   `json:"id"`
	Name      string   `json:"name"`
	Publisher string   `json:"publisher"`
	Summary   string   `json:"summary,omitempty"`
	Tags      []string `json:"tags,omitempty"`
}

// GetModels returns the list of available models from the GitHub catalog
func (c *GitHubClient) GetModels(ctx context.Context) ([]ModelInfo, error) {
	sdkModels, err := c.client.ListModels(ctx)
	if err != nil {
		return nil, err
	}

	var models []ModelInfo
	for _, m := range sdkModels {
		models = append(models, ModelInfo{
			ID:      m.ID,
			Name:    m.Name,
			Summary: m.ID, // Fallback since Family does not exist
		})
	}

	return models, nil
}

// Verify interface compliance
var _ ai.ChatClientInterface = (*GitHubClient)(nil)
