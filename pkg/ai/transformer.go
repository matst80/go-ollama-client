package ai

import (
	"context"
	"strings"
)

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

				// Accumulate tool calls
				if len(chunk.Message.ToolCalls) > 0 {
					for _, tc := range chunk.Message.ToolCalls {
						found := false
						for i, existing := range toolCalls {
							// Match by ID if both have it, or by Index if available
							idMatch := existing.ID != "" && tc.ID != "" && existing.ID == tc.ID
							indexMatch := existing.Index != nil && tc.Index != nil && *existing.Index == *tc.Index

							if idMatch || indexMatch {
								// Update fields if they are provided in this chunk
								if tc.ID != "" {
									toolCalls[i].ID = tc.ID
								}
								if tc.Function.Name != "" {
									toolCalls[i].Function.Name = tc.Function.Name
								}
								if tc.Type != "" {
									toolCalls[i].Type = tc.Type
								}

								// Accumulate arguments
								if len(tc.Function.Arguments) > 0 {
									if indexMatch {
										// log.Printf("appending %s to %s", toolCalls[i].Function.Arguments, tc.Function.Arguments)
										toolCalls[i].Function.Arguments = append(toolCalls[i].Function.Arguments, tc.Function.Arguments...)
									} else {
										toolCalls[i].Function.Arguments = tc.Function.Arguments
									}
								}
								found = true
								break
							}
						}
						if !found {
							// Create a copy to avoid modifying the input chunk if needed
							// (though here it's already a new ToolCall object)
							toolCalls = append(toolCalls, tc)
						}
					}
				}

				// Send the accumulated response
				// We create a copy to handle temporary markdown termination without affecting the actual accumulated content
				toSend := AccumulatedResponse{
					Chunk:            chunk,
					Content:          content.String(),
					ReasoningContent: thinking.String(),
					ToolCalls:        toolCalls,
				}

				// Temporarily terminate unterminated markdown code blocks for better live-preview
				if strings.Count(toSend.Content, "```")%2 != 0 {
					toSend.Content += "\n```"
				}
				if strings.Count(toSend.ReasoningContent, "```")%2 != 0 {
					toSend.ReasoningContent += "\n```"
				}

				output <- &toSend
			}
		}
	}()

	return output
}
