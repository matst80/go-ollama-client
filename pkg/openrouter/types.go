package openrouter

import (
	"encoding/json"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// ChatCompletionChunk represents an OpenRouter chat completion chunk
// based on the response format from models like StepFun or DeepSeek.
type ChatCompletionChunk struct {
	ID       string   `json:"id"`
	Object   string   `json:"object"`
	Created  int64    `json:"created"`
	Model    string   `json:"model"`
	Provider string   `json:"provider"`
	Choices  []Choice `json:"choices"`
}

// Choice represents a choice in the completion chunk
type Choice struct {
	Index              int     `json:"index"`
	Delta              Delta   `json:"delta"`
	FinishReason       *string `json:"finish_reason"`
	NativeFinishReason *string `json:"native_finish_reason"`
}

// Delta represents the delta content in the completion chunk
type Delta struct {
	Content          string            `json:"content"`
	Role             string            `json:"role"`
	Reasoning        string            `json:"reasoning,omitempty"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []DeltaToolCall   `json:"tool_calls,omitempty"`
}

// DeltaToolCall represents a tool call in a delta
type DeltaToolCall struct {
	Index    int           `json:"index"`
	ID       string        `json:"id,omitempty"`
	Type     string        `json:"type,omitempty"`
	Function DeltaFunction `json:"function"`
}

// DeltaFunction represents a function call in a delta
type DeltaFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ReasoningDetail represents the reasoning details in the delta
type ReasoningDetail struct {
	Type   string `json:"type"`
	Text   string `json:"text"`
	Format string `json:"format"`
	Index  int    `json:"index"`
}

// ToChatResponse converts a ChatCompletionChunk to an ai.ChatResponse.
// It maps OpenRouter specific fields like reasoning to the unified ai.Message structure.
func (c *ChatCompletionChunk) ToChatResponse() *ai.ChatResponse {
	if len(c.Choices) == 0 {
		return &ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{
				Model:     c.Model,
				CreatedAt: time.Unix(c.Created, 0),
			},
		}
	}

	choice := c.Choices[0]

	// Map role, defaulting to assistant for delta chunks
	role := ai.MessageRoleAssistant
	if choice.Delta.Role != "" {
		role = ai.MessageRole(choice.Delta.Role)
	}

	msg := ai.Message{
		Role:             role,
		Content:          choice.Delta.Content,
		ReasoningContent: choice.Delta.Reasoning,
	}

	// If reasoning is empty but reasoning_details has content, accumulate it
	if msg.ReasoningContent == "" && len(choice.Delta.ReasoningDetails) > 0 {
		for _, detail := range choice.Delta.ReasoningDetails {
			msg.ReasoningContent += detail.Text
		}
	}

	// Handle ToolCalls
	if len(choice.Delta.ToolCalls) > 0 {
		for _, dtc := range choice.Delta.ToolCalls {
			idx := dtc.Index
			msg.ToolCalls = append(msg.ToolCalls, ai.ToolCall{
				Index: &idx,
				ID:    dtc.ID,
				Type:  dtc.Type,
				Function: ai.FunctionCall{
					Name:      dtc.Function.Name,
					Arguments: json.RawMessage(dtc.Function.Arguments),
				},
			})
		}
	}

	resp := &ai.ChatResponse{
		BaseResponse: &ai.BaseResponse{
			Model:     c.Model,
			CreatedAt: time.Unix(c.Created, 0),
		},
		Message: msg,
	}

	if choice.FinishReason != nil {
		resp.Done = true
		resp.DoneReason = *choice.FinishReason
	}

	return resp
}

// --- Strongly typed OpenRouter request structures and helper conversion --- //

// OpenRouterFunction models the function call structure expected by OpenRouter:
// `arguments` must be a JSON string (i.e., a string containing JSON text).
type OpenRouterFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// OpenRouterToolCall models a tool call inside a message for OpenRouter
type OpenRouterToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function OpenRouterFunction `json:"function"`
}

// OpenRouterMessage is the message shape for OpenRouter requests
type OpenRouterMessage struct {
	Role             ai.MessageRole       `json:"role"`
	Content          string               `json:"content"`
	ReasoningContent string               `json:"thinking,omitempty"`
	Images           []string             `json:"images,omitempty"`
	ToolCalls        []OpenRouterToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string               `json:"tool_call_id,omitempty"`
}

// OpenRouterChatRequest is the strongly-typed request shape to send to OpenRouter
type OpenRouterChatRequest struct {
	Model    string               `json:"model"`
	Stream   bool                 `json:"stream"`
	Format   *ai.ResponseFormat   `json:"format,omitempty"`
	Think    any                  `json:"think,omitempty"`
	Messages []OpenRouterMessage  `json:"messages"`
	Tools    []ai.Tool            `json:"tools,omitempty"`
}

// ToOpenRouterChatRequest converts a typed ai.ChatRequest into a strongly-typed OpenRouterChatRequest.
// This ensures that any function arguments (json.RawMessage) are converted into
// a JSON text string as required by OpenRouter/OpenAI-compatible APIs.
func ToOpenRouterChatRequest(req *ai.ChatRequest) OpenRouterChatRequest {
	orReq := OpenRouterChatRequest{
		Model:    req.Model,
		Stream:   req.Stream,
		Format:   req.Format,
		Think:    req.Think,
		Messages: make([]OpenRouterMessage, 0, len(req.Messages)),
		Tools:    req.Tools,
	}

	for _, m := range req.Messages {
		orm := OpenRouterMessage{
			Role:             m.Role,
			Content:          m.Content,
			ReasoningContent: m.ReasoningContent,
			Images:           m.Images,
			ToolCallID:       m.ToolCallID,
		}

		if len(m.ToolCalls) > 0 {
			orm.ToolCalls = make([]OpenRouterToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				orf := OpenRouterFunction{
					Name: tc.Function.Name,
				}

				if len(tc.Function.Arguments) > 0 {
					orf.Arguments = string(tc.Function.Arguments)
				}

				idx := tc.Index
				orm.ToolCalls = append(orm.ToolCalls, OpenRouterToolCall{
					Index:    idx,
					ID:       tc.ID,
					Type:     tc.Type,
					Function: orf,
				})
			}
		}

		orReq.Messages = append(orReq.Messages, orm)
	}

	return orReq
}
