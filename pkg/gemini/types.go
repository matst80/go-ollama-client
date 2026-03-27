package gemini

import (
	"encoding/json"
	"fmt"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"google.golang.org/genai"
)

// Helper to convert ai.ChatRequest to []*genai.Content and *genai.GenerateContentConfig
func ToGeminiRequest(req ai.ChatRequest) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	contents := make([]*genai.Content, 0)
	var systemInstr *genai.Content

	for _, msg := range req.Messages {
		if msg.Role == ai.MessageRoleSystem {
			systemInstr = &genai.Content{
				Parts: []*genai.Part{{Text: msg.Content}},
			}
			continue
		}

		role := "user"
		if msg.Role == ai.MessageRoleAssistant {
			role = "model"
		}

		parts := make([]*genai.Part, 0)
		if msg.Content != "" && msg.Role != ai.MessageRoleTool {
			parts = append(parts, &genai.Part{Text: msg.Content})
		}

		for _, tc := range msg.ToolCalls {
			var args map[string]any
			json.Unmarshal(tc.Function.Arguments, &args)
			part := &genai.Part{
				FunctionCall: &genai.FunctionCall{
					Name: tc.Function.Name,
					Args: args,
				},
			}
			if tc.ThoughtSignature != "" {
				part.ThoughtSignature = []byte(tc.ThoughtSignature)
			}
			parts = append(parts, part)
		}

		if msg.Role == ai.MessageRoleTool {
			var response map[string]any
			if err := json.Unmarshal([]byte(msg.Content), &response); err != nil {
				response = map[string]any{"result": msg.Content}
			}
			// Note: We try to use the ToolCallID as the name if we don't have the function name,
			// though Gemini prefers the function name. If the previous message was a model call,
			// we could try to look it up, but for now we'll stick to ID.
			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name:     msg.ToolCallID,
					Response: response,
				},
			})
		}

		// Merge with previous message if role is the same
		if len(contents) > 0 && contents[len(contents)-1].Role == role {
			contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
		} else if len(parts) > 0 {
			contents = append(contents, &genai.Content{
				Role:  role,
				Parts: parts,
			})
		}
	}

	config := &genai.GenerateContentConfig{
		SystemInstruction: systemInstr,
	}

	if len(req.Tools) > 0 {
		funcs := make([]*genai.FunctionDeclaration, 0)
		var googleSearch *genai.GoogleSearchRetrieval

		for _, t := range req.Tools {
			if t.Function.Name == "google_search_retrieval" || t.Function.Name == "google_search" {
				googleSearch = &genai.GoogleSearchRetrieval{}
				continue
			}

			// Convert json schema in function parameters to genai.Schema
			var paramsSchema *genai.Schema
			sanitizedParams := sanitizeSchema(t.Function.Parameters)
			b, err := json.Marshal(sanitizedParams)
			if err == nil {
				json.Unmarshal(b, &paramsSchema)
			}

			funcs = append(funcs, &genai.FunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  paramsSchema,
			})
		}

		if len(funcs) > 0 {
			config.Tools = append(config.Tools, &genai.Tool{
				FunctionDeclarations: funcs,
			})
		}
		if googleSearch != nil {
			config.Tools = append(config.Tools, &genai.Tool{
				GoogleSearchRetrieval: googleSearch,
			})
		}
	}

	if req.Format != nil {
		if *req.Format == ai.ResponseFormatJson {
			config.ResponseMIMEType = "application/json"
		}
	}

	return contents, config, nil
}

// sanitizeSchema recursively ensures that all array types in the JSON schema have an "items" field,
// which is required by the Gemini API.
func sanitizeSchema(schema any) any {
	if m, ok := schema.(map[string]any); ok {
		// Fix array type
		if t, ok := m["type"].(string); ok && t == "array" {
			if _, hasItems := m["items"]; !hasItems {
				m["items"] = map[string]any{"type": "string"}
			}
		}

		// Recurse into properties
		if props, ok := m["properties"].(map[string]any); ok {
			for k, v := range props {
				props[k] = sanitizeSchema(v)
			}
		}

		// Recurse into items (if it's an array of objects/arrays)
		if items, ok := m["items"].(map[string]any); ok {
			m["items"] = sanitizeSchema(items)
		}

		return m
	}

	// For standard ai.Tool, parameters might already be from ai.generateJSONSchema (map[string]interface{})
	if m, ok := schema.(map[string]interface{}); ok {
		// Fix array type
		if t, ok := m["type"].(string); ok && t == "array" {
			if _, hasItems := m["items"]; !hasItems {
				m["items"] = map[string]interface{}{"type": "string"}
			}
		}

		// Recurse into properties
		if props, ok := m["properties"].(map[string]interface{}); ok {
			for k, v := range props {
				props[k] = sanitizeSchema(v)
			}
		}

		// Recurse into items
		if items, ok := m["items"].(map[string]interface{}); ok {
			m["items"] = sanitizeSchema(items)
		}

		return m
	}

	return schema
}

// Helper to convert genai.GenerateContentResponse to ai.ChatResponse
func ToChatResponse(gr *genai.GenerateContentResponse) *ai.ChatResponse {
	if gr == nil || len(gr.Candidates) == 0 {
		return &ai.ChatResponse{
			BaseResponse: &ai.BaseResponse{
				Done: true,
			},
		}
	}

	cand := gr.Candidates[0]
	content := ""
	var toolCalls []ai.ToolCall

	if cand.Content != nil {
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				content += part.Text
			}
			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				toolCalls = append(toolCalls, ai.ToolCall{
					ID: part.FunctionCall.ID,
					Function: ai.FunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: args,
					},
					ThoughtSignature: string(part.ThoughtSignature),
				})
			}
		}
	}

	if cand.GroundingMetadata != nil {
		content += "\n\nSources:\n"
		for _, chunk := range cand.GroundingMetadata.GroundingChunks {
			if chunk.Web != nil {
				content += fmt.Sprintf("- [%s](%s)\n", chunk.Web.Title, chunk.Web.URI)
			}
		}
	}

	role := ai.MessageRoleAssistant
	if cand.Content != nil {
		if cand.Content.Role == "user" {
			role = ai.MessageRoleUser
		}
	}

	doneReason := string(cand.FinishReason)

	return &ai.ChatResponse{
		BaseResponse: &ai.BaseResponse{
			Done:       doneReason != "" && doneReason != "FINISH_REASON_UNSPECIFIED",
			DoneReason: doneReason,
		},
		Message: ai.Message{
			Role:      role,
			Content:   content,
			ToolCalls: toolCalls,
		},
	}
}
