package ai

import (
	"context"
	"strings"
)

// mergeToolCalls merges incoming tool calls into the existing slice, updating matching entries by ID or Index.
// It also merges argument-only chunks (no ID and no Index) into the last existing tool call so that
// tool calls split across streaming chunks are assembled into a single complete ToolCall.
func mergeToolCalls(existing []ToolCall, incoming []ToolCall) []ToolCall {
	if len(incoming) == 0 {
		return existing
	}

	for _, tc := range incoming {
		found := false
		for i, ex := range existing {
			// Match by ID if both have it, or by Index if available
			idMatch := ex.ID != "" && tc.ID != "" && ex.ID == tc.ID
			indexMatch := ex.Index != nil && tc.Index != nil && *ex.Index == *tc.Index

			if idMatch || indexMatch {
				// Update fields if they are provided in this chunk
				if tc.ID != "" {
					existing[i].ID = tc.ID
				} else {
					tc.ID = existing[i].ID
				}
				if tc.Function.Name != "" {
					existing[i].Function.Name = tc.Function.Name
				}
				if tc.Type != "" {
					existing[i].Type = tc.Type
				}

				// Accumulate arguments
				if len(tc.Function.Arguments) > 0 {
					if indexMatch {
						// If we matched by index, append partial arguments to support chunked argument delivery
						existing[i].Function.Arguments = append(existing[i].Function.Arguments, tc.Function.Arguments...)
					} else {
						// If matched by ID but not index, replace (this follows previous behavior)
						existing[i].Function.Arguments = tc.Function.Arguments
					}
				}
				found = true
				break
			}
		}

		if !found {
			// Handle argument-only chunks: some providers send the tool call metadata (index/id/name)
			// in one chunk and the raw arguments in a subsequent chunk that omits index/id.
			// If the incoming call lacks ID and Index but contains arguments, merge those arguments
			// into the last existing tool call (the most recent one), instead of appending a new call.
			if tc.ID == "" && tc.Index == nil && len(tc.Function.Arguments) > 0 && len(existing) > 0 {
				last := len(existing) - 1
				// Update name/type if provided in this partial chunk
				if tc.Function.Name != "" {
					existing[last].Function.Name = tc.Function.Name
				}
				if tc.Type != "" {
					existing[last].Type = tc.Type
				}
				// Append arguments to the last tool call to complete the JSON payload
				existing[last].Function.Arguments = append(existing[last].Function.Arguments, tc.Function.Arguments...)
			} else {
				// Append new tool call when no match was found
				existing = append(existing, tc)
			}
		}
	}

	return existing
}

// StreamAccumulator takes a receive-only channel of ChatResponse and returns a receive-only channel of AccumulatedResponse.
// It keeps track of the accumulated message (content, reasoning_content, and tool_calls).
func StreamAccumulator(ctx context.Context, input <-chan *ChatResponse, autoCloseMarkdown bool) <-chan *AccumulatedResponse {
	output := make(chan *AccumulatedResponse)

	go func() {
		defer close(output)

		var content strings.Builder
		var thinking strings.Builder
		toolCalls := make([]ToolCall, 0)

		for {
			select {
			case <-ctx.Done():
				return
			case chunk, ok := <-input:
				if !ok {
					return
				}

				// Accumulate content
				if chunk.Message.Content != "" {
					content.WriteString(chunk.Message.Content)
				}

				// Accumulate reasoning content
				if chunk.Message.ReasoningContent != "" {
					thinking.WriteString(chunk.Message.ReasoningContent)
				}

				// Accumulate tool calls via helper
				if len(chunk.Message.ToolCalls) > 0 {
					toolCalls = mergeToolCalls(toolCalls, chunk.Message.ToolCalls)
				}

				// Send the accumulated response
				// We create a copy to handle temporary markdown termination without affecting the actual accumulated content
				toSend := AccumulatedResponse{
					Chunk:            chunk,
					Content:          content.String(),
					ReasoningContent: thinking.String(),
					ToolCalls:        toolCalls,
				}

				if autoCloseMarkdown {
					if strings.Count(toSend.Content, "```")%2 != 0 {
						toSend.Content += "\n```"
					}
					if strings.Count(toSend.ReasoningContent, "```")%2 != 0 {
						toSend.ReasoningContent += "\n```"
					}
				}

				output <- &toSend
			}
		}
	}()

	return output
}
