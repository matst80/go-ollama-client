package xai

import (
	"encoding/json"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// ChatCompletionChunk represents an xAI chat completion chunk.
// The structure is intentionally similar to OpenRouter's streaming format
// so it can be parsed via the shared DataJsonChunkReader and converted
// into the unified ai.ChatResponse.
type ChatCompletionChunk struct {
	ID       string   `json:"id,omitempty"`
	Object   string   `json:"object,omitempty"`
	Created  int64    `json:"created,omitempty"`
	Model    string   `json:"model,omitempty"`
	Provider string   `json:"provider,omitempty"`
	Choices  []Choice `json:"choices,omitempty"`
}

// Choice represents a choice in the streaming chunk
type Choice struct {
	Index              int     `json:"index,omitempty"`
	Delta              Delta   `json:"delta"`
	FinishReason       *string `json:"finish_reason,omitempty"`
	NativeFinishReason *string `json:"native_finish_reason,omitempty"`
}

// Delta represents the incremental delta content for a choice
type Delta struct {
	Content          string            `json:"content,omitempty"`
	Role             string            `json:"role,omitempty"`
	Reasoning        string            `json:"reasoning,omitempty"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []DeltaToolCall   `json:"tool_calls,omitempty"`
}

// DeltaToolCall represents a tool call present in a delta payload
type DeltaToolCall struct {
	Index    int           `json:"index,omitempty"`
	ID       string        `json:"id,omitempty"`
	Type     string        `json:"type,omitempty"`
	Function DeltaFunction `json:"function"`
}

// DeltaFunction holds the function call details in a delta
type DeltaFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ReasoningDetail is a piece of the model's chain-of-thought / reasoning explanation
type ReasoningDetail struct {
	Type   string `json:"type,omitempty"`
	Text   string `json:"text,omitempty"`
	Format string `json:"format,omitempty"`
	Index  int    `json:"index,omitempty"`
}

// --- Strongly typed XAI request structures and helper conversion --- //

// XAIFunction models the function call structure expected by xAI:
// `arguments` must be a JSON string (i.e., a string containing JSON text).
type XAIFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// XAIToolCall models a tool call inside a message for xAI
type XAIToolCall struct {
	Index    *int        `json:"index,omitempty"`
	ID       string      `json:"id,omitempty"`
	Type     string      `json:"type,omitempty"`
	Function XAIFunction `json:"function"`
}

// XAIMessage is the message shape for xAI requests
type XAIMessage struct {
	Role             ai.MessageRole `json:"role"`
	Content          any            `json:"content"`
	ReasoningContent string         `json:"thinking,omitempty"`
	ToolCalls        []XAIToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

// XAIChatRequest is the strongly-typed request shape to send to xAI
type XAIChatRequest struct {
	Model           string             `json:"model"`
	Stream          bool               `json:"stream"`
	Format          *ai.ResponseFormat `json:"format,omitempty"`
	Think           any                `json:"think,omitempty"`
	Messages        []XAIMessage       `json:"messages"`
	Tools           []ai.Tool          `json:"tools,omitempty"`
	ReasoningEffort string             `json:"reasoning_effort,omitempty"`
}

/*
{
   "deferred": null,
   "frequency_penalty": null,
   "logit_bias": null,
   "logprobs": null,
   "max_completion_tokens": null,
   "max_tokens": null,
   "messages": [],
   "model": "",
   "n": null,
   "parallel_tool_calls": null,
   "presence_penalty": null,
   "reasoning_effort": null,
   "response_format": [
     null,
     [
       {
         "type": ""
       },
       {
         "type": ""
       },
       {
         "json_schema": null,
         "type": ""
       }
     ]
   ],
   "search_parameters": [
     null,
     {
       "from_date": null,
       "max_search_results": null,
       "mode": null,
       "return_citations": null,
       "sources": null,
       "to_date": null
     }
   ],
   "seed": null,
   "stop": null,
   "stream": null,
   "stream_options": [
     null,
     {
       "include_usage": false
     }
   ],
   "temperature": null,
   "tool_choice": [
     null,
     [
       "",
       {
         "function": [
           null,
           {
             "name": ""
           }
         ],
         "type": ""
       }
     ]
   ],
   "tools": null,
   "top_logprobs": null,
   "top_p": null,
   "user": null,
   "web_search_options": [
     null,
     {
       "filters": null,
       "search_context_size": null,
       "user_location": null
     }
   ]
 }
*/

// ToXAIChatRequest converts a typed ai.ChatRequest into a strongly-typed XAIChatRequest.
// This preserves the same mapping logic as the previous generic mapper: it converts
// any function arguments (json.RawMessage) into a JSON text string as required by xAI.
func ToXAIChatRequest(req *ai.ChatRequest) XAIChatRequest {
	xReq := XAIChatRequest{
		Model:    req.Model,
		Stream:   req.Stream,
		Format:   req.Format,
		Think:    req.Think,
		Messages: make([]XAIMessage, 0, len(req.Messages)),
		Tools:    req.Tools,
	}
	switch req.Think {
	case "high":
		xReq.ReasoningEffort = "high"
	case "low":
		xReq.ReasoningEffort = "low"
	}

	for _, m := range req.Messages {
		xm := XAIMessage{
			Role:             m.Role,
			ReasoningContent: m.ReasoningContent,
			ToolCallID:       m.ToolCallID,
		}

		if len(m.Images) > 0 {
			parts := make([]map[string]interface{}, 0, len(m.Images)+1)
			if m.Content != "" {
				parts = append(parts, map[string]interface{}{
					"type": "text",
					"text": m.Content,
				})
			}
			for _, img := range m.Images {
				parts = append(parts, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": img,
					},
				})
			}
			xm.Content = parts
		} else {
			xm.Content = m.Content
		}

		if len(m.ToolCalls) > 0 {
			xm.ToolCalls = make([]XAIToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				xf := XAIFunction{
					Name: tc.Function.Name,
				}

				if len(tc.Function.Arguments) > 0 {
					xf.Arguments = string(tc.Function.Arguments)
				}

				idx := tc.Index
				xm.ToolCalls = append(xm.ToolCalls, XAIToolCall{
					Index:    idx,
					ID:       tc.ID,
					Type:     tc.Type,
					Function: xf,
				})
			}
		}

		xReq.Messages = append(xReq.Messages, xm)
	}

	return xReq
}

// ToChatResponse converts an xAI ChatCompletionChunk into the unified ai.ChatResponse
// The conversion follows the same approach as the OpenRouter adapter: the first choice
// is used (streaming typically only returns a single streaming choice) and delta fields
// are mapped into ai.Message. Tool calls and reasoning details are also converted.
func (c *ChatCompletionChunk) ToChatResponse() *ai.ChatResponse {
	// If there are no choices, return a minimal response with base fields
	if len(c.Choices) == 0 {
		return &ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{
				Model:     c.Model,
				CreatedAt: time.Unix(c.Created, 0),
			},
		}
	}

	choice := c.Choices[0]

	// Default role to assistant when not provided
	role := ai.MessageRoleAssistant
	if choice.Delta.Role != "" {
		role = ai.MessageRole(choice.Delta.Role)
	}

	msg := ai.Message{
		Role:             role,
		Content:          choice.Delta.Content,
		ReasoningContent: choice.Delta.Reasoning,
	}

	// If ReasoningContent is empty, but ReasoningDetails exist, concatenate them
	if msg.ReasoningContent == "" && len(choice.Delta.ReasoningDetails) > 0 {
		for _, detail := range choice.Delta.ReasoningDetails {
			msg.ReasoningContent += detail.Text
		}
	}

	// Convert tool calls if present
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
