package ai

import (
	"context"
)

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
