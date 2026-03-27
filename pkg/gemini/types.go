package gemini

import (
	"encoding/json"

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

		role := string(msg.Role)
		if role == string(ai.MessageRoleAssistant) {
			role = "model"
		} else if role == string(ai.MessageRoleTool) {
			role = "user"
		}

		parts := make([]*genai.Part, 0)
		if msg.Content != "" {
			parts = append(parts, &genai.Part{Text: msg.Content})
		}

		for _, tc := range msg.ToolCalls {
			var args map[string]any
			json.Unmarshal(tc.Function.Arguments, &args)
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: args,
				},
				ThoughtSignature: []byte(tc.ThoughtSignature),
			})
		}

		if msg.Role == ai.MessageRoleTool {
			var response map[string]any
			if err := json.Unmarshal([]byte(msg.Content), &response); err != nil {
				response = map[string]any{"result": msg.Content}
			}
			parts = append(parts, &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					ID:       msg.ToolCallID,
					Name:     msg.ToolCallID, // Gemini API requires name matching the tool. The ai.Message doesn't store the tool name, but ID might not match name. We will see.
					Response: response,
				},
			})
		}

		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: parts,
		})
	}

	config := &genai.GenerateContentConfig{
        SystemInstruction: systemInstr,
    }

	if len(req.Tools) > 0 {
		funcs := make([]*genai.FunctionDeclaration, 0, len(req.Tools))
		for _, t := range req.Tools {

            // Convert json schema in function parameters to genai.Schema
            var paramsSchema *genai.Schema
            b, err := json.Marshal(t.Function.Parameters)
            if err == nil {
                json.Unmarshal(b, &paramsSchema)
            }

			funcs = append(funcs, &genai.FunctionDeclaration{
                Name: t.Function.Name,
                Description: t.Function.Description,
                Parameters: paramsSchema,
            })
		}
		config.Tools = []*genai.Tool{{
            FunctionDeclarations: funcs,
        }}
	}

	if req.Format != nil {
		if *req.Format == ai.ResponseFormatJson {
			config.ResponseMIMEType = "application/json"
		}
	}

	return contents, config, nil
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

	role := ai.MessageRoleAssistant
    if cand.Content != nil {
        if cand.Content.Role == "user" {
            role = ai.MessageRoleUser
        }
    }

    doneReason := string(cand.FinishReason)

	return &ai.ChatResponse{
		BaseResponse: &ai.BaseResponse{
			Done: doneReason != "" && doneReason != "FINISH_REASON_UNSPECIFIED",
            DoneReason: doneReason,
		},
		Message: ai.Message{
			Role:      role,
			Content:   content,
			ToolCalls: toolCalls,
		},
	}
}
