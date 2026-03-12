package ollama

import (
	"time"

	"github.com/matst80/go-ollama-client/pkg/ai"
)

type ListModelsResponse struct {
	Models []BaseModel `json:"models"`
}

type ListRunningModelsResponse struct {
	Models []RunningModel `json:"models"`
}

type BaseModel struct {
	Name    string  `json:"name"`
	Model   string  `json:"model"`
	Size    int64   `json:"size"`
	Digest  string  `json:"digest"`
	Details Details `json:"details"`
}

type RunningModel struct {
	*BaseModel
	ExpiresAt     time.Time `json:"expires_at"`
	SizeVRAM      int64     `json:"size_vram"`
	ContextLength int       `json:"context_length"`
}

type Details struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type ModelDetailsRequest struct {
	Model   string `json:"model"`
	Verbose *bool  `json:"verbose,omitempty"`
}

type ModelDetailsResponse struct {
	Parameters   string         `json:"parameters"`
	License      string         `json:"license"`
	Capabilities []string       `json:"capabilities"`
	ModifiedAt   time.Time      `json:"modified_at"`
	Details      Details        `json:"details"`
	ModelInfo    map[string]any `json:"model_info,omitempty"`
}

// ModelOptions represents additional parameters for the model
type ModelOptions struct {
	// Temperature is the creativity of the model. Higher is more creative. (Default: 0.8)
	Temperature float64 `json:"temperature,omitempty"`
	// ContextWindowSize sets the size of the context window. (Default: 2048)
	ContextWindowSize int `json:"num_ctx,omitempty"`
	// RepeatPenalty penalizes repetitions. Higher is less repetitive. (Default: 1.1)
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
	// TopP leads to more diverse or focused text. (Default: 0.9)
	TopP float64 `json:"top_p,omitempty"`
	// TopK reduces the probability of generating nonsense. (Default: 40)
	TopK int `json:"top_k,omitempty"`
	// NumPredict is the maximum number of tokens to predict. (Default: 128)
	NumPredict int `json:"num_predict,omitempty"`
	// Seed is the random seed for reproducibility. (Default: 0)
	Seed *int64 `json:"seed,omitempty"`
	// Stop is a list of stop sequences that will halt generation when encountered. (Default: none)
	Stop []string `json:"stop,omitempty"`
	// Minimum probability threshold for token selection
	MinP float64 `json:"min_p,omitempty"`
}

type OllamaChatRequest struct {
	*ai.BaseRequest[OllamaChatRequest]
	// KeepAlive controls how long the model will stay loaded into memory (default: 5m)
	KeepAlive *string `json:"keep_alive,omitempty"`
	// Options are additional model parameters like temperature
	Options *ModelOptions `json:"options,omitempty"`
	// Logprobs if true, return the log probabilities of the tokens
	Logprobs *bool `json:"logprobs,omitempty"`
	// TopLogprobs is the number of top log probabilities to return
	TopLogprobs *int `json:"top_logprobs,omitempty"`
}

func (r *OllamaChatRequest) WithOptions(options *ModelOptions) *OllamaChatRequest {
	r.Options = options
	return r
}

func (r *OllamaChatRequest) WithOptionsBuilder(builder func(*ModelOptions)) *OllamaChatRequest {
	if r.Options == nil {
		r.Options = &ModelOptions{}
	}
	builder(r.Options)
	return r
}

func (r *OllamaChatRequest) WithKeepAlive(keepAlive string) *OllamaChatRequest {
	r.KeepAlive = &keepAlive
	return r
}

func (r *OllamaChatRequest) WithLogProbabilities(logprobs bool) *OllamaChatRequest {
	r.Logprobs = &logprobs
	return r
}

func (r *OllamaChatRequest) WithTopLogProbabilities(topLogprobs int) *OllamaChatRequest {
	r.TopLogprobs = &topLogprobs
	return r
}

// CreateRequest represents the request body for Ollama create API
type CreateRequest struct {
	// Model is the name of the model to use (required)
	Model string `json:"model"`
	// Stream if false, the response will be returned as a single response object
	Stream bool `json:"stream"`
	// From is the name of an existing model to create from (optional)
	From string `json:"from,omitempty"`
	// Modelfile is the contents of the Modelfile (optional)
	Modelfile string `json:"modelfile,omitempty"`
	// Path is the path to the Modelfile (optional)
	Path string `json:"path,omitempty"`
	// Quantize is the optional quantization level (e.g., "q4_0")
	Quantize string `json:"quantize,omitempty"`
}

// NewCreateRequest creates a new CreateRequest with the given model name
func NewCreateRequest(model string) *CreateRequest {
	r := &CreateRequest{Model: model}
	return r
}

func (r *CreateRequest) WithStreaming(stream bool) *CreateRequest {
	r.Stream = stream
	return r
}

func (r *CreateRequest) WithFrom(from string) *CreateRequest {
	r.From = from
	return r
}

func (r *CreateRequest) WithModelfile(modelfile string) *CreateRequest {
	r.Modelfile = modelfile
	return r
}

func (r *CreateRequest) WithPath(path string) *CreateRequest {
	r.Path = path
	return r
}

func (r *CreateRequest) WithQuantize(quantize string) *CreateRequest {
	r.Quantize = quantize
	return r
}

// CreateResponse represents the response from the Ollama create API
type CreateResponse struct {
	Status string `json:"status"`
}

type VersionResponse struct {
	Version string `json:"version"`
}

type CopyRequest struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

type PushPullRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure,omitempty"`
	Stream   bool   `json:"stream"`
}

func NewPushPullRequest(model string) *PushPullRequest {
	return &PushPullRequest{
		Model: model,
	}
}

func (r *PushPullRequest) WithInsecure(insecure bool) *PushPullRequest {
	r.Insecure = insecure
	return r
}

func (r *PushPullRequest) WithStreaming(stream bool) *PushPullRequest {
	r.Stream = stream
	return r
}

type StatusResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
	// Error     string `json:"error,omitempty"`
}

type DeleteRequest struct {
	Model string `json:"model"`
}
