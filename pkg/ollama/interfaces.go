package ollama

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

type ModelClient interface {
	ListModels(ctx context.Context) (*ListModelsResponse, error)
	ListRunningModels(ctx context.Context) (*ListRunningModelsResponse, error)
	ShowModelDetails(ctx context.Context, model string) (*ModelDetailsResponse, error)
}

type ModelModificationClient interface {
	CreateModel(ctx context.Context, req CreateRequest) (*CreateResponse, error)
	CreateModelStreamed(ctx context.Context, req CreateRequest, ch chan *CreateResponse) error
	CopyModel(ctx context.Context, source, destination string) error
	PullModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error)
	PullModelStreamed(ctx context.Context, req PushPullRequest, ch chan *StatusResponse) error
	PushModel(ctx context.Context, req PushPullRequest) (*StatusResponse, error)
	PushModelStreamed(ctx context.Context, req PushPullRequest, ch chan *StatusResponse) error
	DeleteModel(ctx context.Context, model string) error
}

type VersionClient interface {
	GetVersion(ctx context.Context) (string, error)
}
