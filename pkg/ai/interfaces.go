package ai

import (
	"context"
)

// SessionHooks defines a set of callbacks that can be used to monitor and
// react to events within an AgentSession.
type SessionHooks interface {
	// OnChatRequest is called before a chat request is sent to the client.
	// If it returns an error, the request is aborted.
	OnChatRequest(ctx context.Context, req *ChatRequest) error

	// OnThinking is called when a model's thinking process (reasoning content) is streamed.
	// The thinking string contains the delta of the reasoning content.
	OnThinking(ctx context.Context, thinking string)

	// OnContent is called when the model's main content is streamed.
	// The content string contains the delta of the content.
	OnContent(ctx context.Context, content string)

	// OnBeforeToolCall is called after the assistant suggests tool calls but before they are executed.
	// If it returns an error, tool execution is aborted.
	OnBeforeToolCall(ctx context.Context, toolCalls []ToolCall) error

	// OnAfterToolCall is called after tool calls have been executed and their results are available.
	OnAfterToolCall(ctx context.Context, toolCalls []ToolCall, messages []Message, results []AutoToolResult)

	// OnBlock is called when a complete fenced block (e.g., git diff) is parsed from the stream.
	OnBlock(ctx context.Context, blockType string, content string)

	// OnDone is called when the chat turn is completely finished.
	OnDone(ctx context.Context, res AccumulatedResponse)

	// OnFlush is called when the current stream should be flushed to the UI
	// even if the turn is not finished.
	OnFlush(ctx context.Context)

	// OnError is called when an error occurs during the session (e.g., API failure).
	OnError(ctx context.Context, err error)
}

// DefaultSessionHooks provides empty implementations for all SessionHooks methods,
// allowing clients to embed it and only override the hooks they care about.
type DefaultSessionHooks struct{}

func (h *DefaultSessionHooks) OnChatRequest(ctx context.Context, req *ChatRequest) error { return nil }
func (h *DefaultSessionHooks) OnThinking(ctx context.Context, thinking string)           {}
func (h *DefaultSessionHooks) OnContent(ctx context.Context, content string)            {}
func (h *DefaultSessionHooks) OnBeforeToolCall(ctx context.Context, toolCalls []ToolCall) error {
	return nil
}
func (h *DefaultSessionHooks) OnAfterToolCall(ctx context.Context, toolCalls []ToolCall, messages []Message, results []AutoToolResult) {
}
func (h *DefaultSessionHooks) OnBlock(ctx context.Context, blockType string, content string) {}
func (h *DefaultSessionHooks) OnDone(ctx context.Context, res AccumulatedResponse)           {}
func (h *DefaultSessionHooks) OnFlush(ctx context.Context)                          {}
func (h *DefaultSessionHooks) OnError(ctx context.Context, err error)                       {}


// ChatClientInterface defines a minimal, testable abstraction over the concrete
// Client implementation. Use this in production code and tests to decouple
// callers from the concrete `Client` type.
type ChatClientInterface interface {

	// Chat performs a non-streaming request and returns the full ChatResponse.
	Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error)

	// ChatStreamed performs a streaming request and delivers chunks to the provided channel.
	ChatStreamed(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error
}

type GenerateClient interface {
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)
}

type EmbeddingClient interface {
	GenerateEmbeddings(ctx context.Context, req EmbeddingsRequest) (*EmbeddingsResponse, error)
}

type VersionClient interface {
	GetVersion(ctx context.Context) (string, error)
}
