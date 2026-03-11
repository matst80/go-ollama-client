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

		content := ""
		thinking := ""
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
					content += chunk.Message.Content
				}

				// Accumulate reasoning content
				if chunk.Message.ReasoningContent != "" {
					thinking += chunk.Message.ReasoningContent
				}

				// Accumulate tool calls
				if len(chunk.Message.ToolCalls) > 0 {
					// For tool calls, we might need a more sophisticated merge logic
					// if they are streamed partially. Ollama usually sends them as whole objects
					// but let's be safe and track by ID if available.
					for _, tc := range chunk.Message.ToolCalls {
						found := false
						for i, existing := range toolCalls {
							if existing.ID == tc.ID && tc.ID != "" {
								// Merge arguments - this assumes they might be partial
								// but json.RawMessage append isn't always valid JSON.
								// However, if Ollama sends them as chunks, it's usually Content.
								// If they are in ToolCalls, let's just append for now.
								toolCalls[i].Function.Arguments = append(existing.Function.Arguments, tc.Function.Arguments...)
								found = true
								break
							}
						}
						if !found {
							toolCalls = append(toolCalls, tc)
						}
					}
				}

				// Send the accumulated response
				// We create a copy to handle temporary markdown termination without affecting the actual accumulated content
				toSend := AccumulatedResponse{
					Chunk:            chunk,
					Content:          content,
					ReasoningContent: thinking,
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
