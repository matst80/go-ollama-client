package ollama

import "context"

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
