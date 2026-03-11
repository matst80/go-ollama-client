package main

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/matst80/go-ollama-client/pkg/ollama"
	"github.com/matst80/go-ollama-client/pkg/tools"
)

var (
	reasoningStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	infoStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
)

type RunArgs struct {
	Command string `json:"command" tool:"Command to run,required"`
}

func RunCommand(args RunArgs) string {
	fmt.Printf("Running command: %s\n", args.Command)
	command := args.Command
	if command == "" {
		return "no command provided?"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash")
	cmd.Stdin = strings.NewReader(command)

	output, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return string(output) + "\nError: Command timed out after 60s"
	}
	if err != nil {
		return string(output) + "\nError: " + err.Error()
	}
	return string(output)
}

func main() {
	client := ollama.NewOllamaClient("http://localhost:11434")
	ctx := context.Background()
	models, err := client.ListModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, model := range models.Models {
		fmt.Printf("%s, (%d MB)\n", model.Model, model.Size/1024/1024)
	}
	registry := tools.NewRegistry()
	registry.Register("run", &RunArgs{}, RunCommand)

	req := ollama.NewChatRequest("qwen3.5:4b").
		WithStreaming(true).
		WithThinking(true).
		WithOptions(&ollama.ModelOptions{
			ContextWindowSize: 8192,
		}).
		WithTools(registry.GetTools())

	// Initialize the new AgentSession with the request
	agentSession := ollama.NewAgentSession(ctx, client, req, ollama.WithAccumulator())
	defer agentSession.Stop()

	// Tool executor
	executor := tools.NewToolExecutor(registry)

	// Setup renderer - disabled word wrap for cleaner live-overwrite
	renderer, err := glamour.NewTermRenderer(
		glamour.WithAutoStyle(),
		glamour.WithWordWrap(120),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Send the initial user message. This starts the processing loop transparently.
	if err := agentSession.SendUserMessage(ctx, "can you use tool run to get my free disk space?"); err != nil {
		fmt.Printf("\nError: %v\n", err)
	}

	fmt.Println(infoStyle.Render("Streaming Ollama response..."))
	fmt.Println()

	var lastLines []string

	// Simple procedural loop with live overwrite reading from AgentSession's global channel
	for res := range agentSession.Recv() {
		// If the model emitted tool calls, run them (but only once per call ID)
		// Waiting until Done means we have the full chunk, eliminating duplicated tool executions
		if res.Chunk.Done && len(res.ToolCalls) > 0 {
			results, err := executor.HandleCalls(res.ToolCalls)
			if err != nil {
				fmt.Printf("Tool execution error: %v\n", err)
			}

			// Deliver tool execution results back into the session
			var resultMsgs []ollama.Message
			for _, tr := range results {
				msg := tr.ToResultMessage()
				resultMsgs = append(resultMsgs, *msg)
				fmt.Printf("\n[tool %s] %s\n", msg.ToolCallID, msg.Content)
			}

			if len(resultMsgs) > 0 {
				// We append the result msgs explicitly and start another streaming response
				if err := agentSession.SendMessages(ctx, resultMsgs...); err != nil {
					fmt.Printf("failed to deliver tool results: %v\n", err)
				}
			}
		}

		// Build the output string
		var output strings.Builder
		if res.ReasoningContent != "" {
			output.WriteString(reasoningStyle.Render("Thinking:\n"))
			reasoned, err := renderer.Render(res.ReasoningContent)
			if err != nil {
				output.WriteString(err.Error())
			}
			output.WriteString(reasoned)
			output.WriteString("\n")
		}

		if res.Content != "" {
			rendered, err := renderer.Render(res.Content)
			if err != nil {
				output.WriteString(err.Error())
			}
			output.WriteString(rendered)
		}

		outStr := output.String()
		if outStr == "" {
			continue
		}
		lines := strings.Split(strings.TrimRight(outStr, "\n"), "\n")

		// Find the first line that differs
		diffLine := 0
		for diffLine < len(lines) && diffLine < len(lastLines) && lines[diffLine] == lastLines[diffLine] {
			diffLine++
		}

		// If everything is the same and length is same, skip
		if diffLine == len(lines) && len(lines) == len(lastLines) {
			if res.Chunk.Done && len(res.ToolCalls) == 0 {
				break
			}
			continue
		}

		// Move up and clear only if we have previous output
		if len(lastLines) > 0 {
			moveUp := len(lastLines) - diffLine
			if moveUp > 0 {
				fmt.Printf("\033[%dA\r\033[J", moveUp)
			}
		}

		// Print from the first changed line
		for i := diffLine; i < len(lines); i++ {
			fmt.Println(lines[i])
		}
		lastLines = lines

		// Only break loop if the conversation logic allows
		if res.Chunk.Done && len(res.ToolCalls) == 0 {
			break
		}
	}
}
