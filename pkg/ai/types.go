package ai

import (
	"encoding/json"
	"time"
)

type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleTool      MessageRole = "tool"
)

// Message represents a chat message in the Ollama API
type Message struct {
	// Role is the role of the message (system, user, assistant, or tool)
	Role MessageRole `json:"role"`
	// Content is the text content of the message
	Content string `json:"content"`
	// ReasoningContent is the thinking process of the model (for models that support it)
	ReasoningContent string `json:"thinking,omitempty"`
	// Images is an optional list of base64-encoded images for multimodal models
	Images []string `json:"images,omitempty"`
	// ToolCalls are the tool calls made by the model
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is the ID of the tool call this message is responding to (role must be tool)
	ToolCallID string `json:"tool_call_id,omitempty"`
}

func NewMessage(role MessageRole, content string) *Message {
	return &Message{
		Role:    role,
		Content: content,
	}
}

func NewToolResponseMessage(toolCallID string, content string) *Message {
	return &Message{
		Role:       MessageRoleTool,
		ToolCallID: toolCallID,
		Content:    content,
	}
}

func (m *Message) SetImages(images []string) *Message {
	m.Images = images
	return m
}

func (m *Message) SetToolCalls(toolCalls []ToolCall) *Message {
	m.ToolCalls = toolCalls
	return m
}

func (m *Message) SetToolCallID(toolCallID string) *Message {
	m.ToolCallID = toolCallID
	return m
}

type ToolType string

const (
	ToolTypeFunction ToolType = "function"
)

// Tool represents a tool that the AI can use
type Tool struct {
	Type     ToolType `json:"type"`
	Function Function `json:"function"`
}

// Function represents the function definition for a tool
type Function struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// ToolCall represents a call to a tool from the AI
type ToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Logprob represents token log probabilities
type Logprob struct {
	Token       string    `json:"token"`
	Logprob     float64   `json:"logprob"`
	Bytes       []int     `json:"bytes"`
	TopLogprobs []Logprob `json:"top_logprobs,omitempty"`
}

type ResponseFormat string

const (
	ResponseFormatJson ResponseFormat = "json"
	ResponseFormatText ResponseFormat = "text"
)

type ThinkingLevel string

const (
	ThinkingHigh   ThinkingLevel = "high"
	ThinkingMedium ThinkingLevel = "medium"
	ThinkingLow    ThinkingLevel = "low"
)

// BaseRequest contains common fields for all Ollama requests
type BaseRequest[T any] struct {
	parent *T

	// Model is the name of the model to use (required)
	Model string `json:"model"`
	// Stream if false, the response will be returned as a single response object
	Stream bool `json:"stream"`
	// Format is the format to return a response in (e.g., "json")
	Format *ResponseFormat `json:"format,omitempty"`

	// Think if true, the model will return its thinking process
	Think any `json:"think,omitempty"`
}

func (r *BaseRequest[T]) WithModel(model string) *T {
	r.Model = model
	return r.parent
}

func (r *BaseRequest[T]) WithStreaming(stream bool) *T {
	r.Stream = stream
	return r.parent
}

func (r *BaseRequest[T]) WithFormat(format ResponseFormat) *T {
	r.Format = &format
	return r.parent
}

func (r *BaseRequest[T]) WithThinking(think bool) *T {
	r.Think = think
	return r.parent
}

func (r *BaseRequest[T]) WithThinkingLevel(think ThinkingLevel) *T {
	r.Think = think
	return r.parent
}

// BaseResponse contains common fields for all Ollama responses
type BaseResponse struct {
	Model              string     `json:"model"`
	CreatedAt          time.Time  `json:"created_at"`
	Done               bool       `json:"done"`
	DoneReason         string     `json:"done_reason,omitempty"`
	TotalDuration      int64      `json:"total_duration,omitempty"`
	LoadDuration       int64      `json:"load_duration,omitempty"`
	PromptEvalCount    int        `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64      `json:"prompt_eval_duration,omitempty"`
	EvalCount          int        `json:"eval_count,omitempty"`
	EvalDuration       int64      `json:"eval_duration,omitempty"`
	Logprobs           *[]Logprob `json:"logprobs,omitempty"`
	Error              *string    `json:"error,omitempty"`
}

// ChatRequest represents the request body for Ollama chat API
type ChatRequest struct {
	BaseRequest[ChatRequest]
	// Messages are the messages of the conversation, used for chat history (required)
	Messages []Message `json:"messages"`
	// Tools are the tools the model may use
	Tools []Tool `json:"tools,omitempty"`
}

// NewChatRequest creates a new ChatRequest with the given model
func NewChatRequest(model string) *ChatRequest {
	r := &ChatRequest{}
	r.parent = r
	r.Model = model
	return r
}

// AddMessage adds a simple message to the request
func (r *ChatRequest) AddMessage(role MessageRole, content string) *ChatRequest {
	r.Messages = append(r.Messages, Message{
		Role:    role,
		Content: content,
	})
	return r
}

// AddMessageStruct adds a full message struct to the request
func (r *ChatRequest) AddMessageStruct(msg *Message) *ChatRequest {
	r.Messages = append(r.Messages, *msg)
	return r
}

// AddTool adds a tool to the request
func (r *ChatRequest) AddTool(tool Tool) *ChatRequest {
	r.Tools = append(r.Tools, tool)
	return r
}

func (r *ChatRequest) WithTools(tools []Tool) *ChatRequest {
	r.Tools = tools
	return r
}

// ChatResponse represents the response chunk
type ChatResponse struct {
	*BaseResponse
	Message Message `json:"message"`
}

// AccumulatedResponse represents a stream chunk paired with the fully accumulated message
type AccumulatedResponse struct {
	Chunk            *ChatResponse
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCall
}

// GenerateRequest represents the request body for Ollama generate API
type GenerateRequest struct {
	BaseRequest[GenerateRequest]
	// Prompt is the prompt to generate a response for (required)
	Prompt string `json:"prompt"`
	// Suffix is the text after the model response (optional)
	Suffix string `json:"suffix,omitempty"`
	// Images is an optional list of base64-encoded images for multimodal models
	Images []string `json:"images,omitempty"`
	// System is the system prompt (optional)
	System string `json:"system,omitempty"`
	// Template is the prompt template to use (optional)
	Template string `json:"template,omitempty"`
	// Raw if true, no prompt templating will be applied
	Raw bool `json:"raw,omitempty"`
	// Context is the context parameter returned from a previous request (optional)
	Context []int `json:"context,omitempty"`
}

// NewGenerateRequest creates a new GenerateRequest with the given model and prompt
func NewGenerateRequest(model, prompt string) *GenerateRequest {
	r := &GenerateRequest{}
	r.parent = r
	r.Model = model
	r.Prompt = prompt
	return r
}

func (r *GenerateRequest) WithSystem(system string) *GenerateRequest {
	r.System = system
	return r
}

func (r *GenerateRequest) WithContext(context []int) *GenerateRequest {
	r.Context = context
	return r
}

func (r *GenerateRequest) WithSuffix(suffix string) *GenerateRequest {
	r.Suffix = suffix
	return r
}

func (r *GenerateRequest) WithImages(images []string) *GenerateRequest {
	r.Images = images
	return r
}

func (r *GenerateRequest) WithRaw(raw bool) *GenerateRequest {
	r.Raw = raw
	return r
}

// GenerateResponse represents the response from the Ollama generate API
type GenerateResponse struct {
	*BaseResponse
	Response string `json:"response"`
	Thinking string `json:"thinking,omitempty"`
	Context  []int  `json:"context,omitempty"`
}

// EmbeddingsRequest represents the request body for Ollama embeddings API
type EmbeddingsRequest struct {
	BaseRequest[EmbeddingsRequest]
	// Input is the text or array of texts to generate embeddings for (required)
	Input any `json:"input"`
	// Truncate if true, truncate inputs that exceed the context window. If false, returns an error.
	Truncate *bool `json:"truncate,omitempty"`
	// Dimensions is the number of dimensions to generate embeddings for (optional)
	Dimensions *int `json:"dimensions,omitempty"`
}

// NewEmbeddingsRequest creates a new EmbeddingsRequest with the given model and input
func NewEmbeddingsRequest(model string, input any) *EmbeddingsRequest {
	r := &EmbeddingsRequest{}
	r.parent = r
	r.Model = model
	r.Input = input
	return r
}

func (r *EmbeddingsRequest) WithInput(input string) *EmbeddingsRequest {
	r.Input = input
	return r
}

func (r *EmbeddingsRequest) WithInputs(inputs []string) *EmbeddingsRequest {
	r.Input = inputs
	return r
}

func (r *EmbeddingsRequest) WithTruncate(truncate bool) *EmbeddingsRequest {
	r.Truncate = &truncate
	return r
}

func (r *EmbeddingsRequest) WithDimensions(dimensions int) *EmbeddingsRequest {
	r.Dimensions = &dimensions
	return r
}

// EmbeddingsResponse represents the response from the Ollama embeddings API
type EmbeddingsResponse struct {
	*BaseResponse
	Embeddings [][]float64 `json:"embeddings"`
}
