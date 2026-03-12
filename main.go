package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/gemini"
	"github.com/matst80/go-ai-agent/pkg/ollama"
	"github.com/matst80/go-ai-agent/pkg/tools"
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

func ListOllamaModels(client *ollama.OllamaClient) {
	models, err := client.ListModels(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	for _, model := range models.Models {
		fmt.Printf("%s, (%d MB)\n", model.Model, model.Size/1024/1024)
	}
}

func PullModel(client *ollama.OllamaClient, model string) error {
	ch := make(chan *ollama.StatusResponse)
	go func() {
		log.Printf("Pulling %s", model)
		for resp := range ch {
			if resp.Total == 0 {
				continue
			}
			fmt.Printf("\r\033[2KPulling... %3d%%", int(float64(resp.Completed)/float64(resp.Total)*100.0))
		}
		log.Println("Download completed")
	}()

	return client.PullModelStreamed(context.Background(), *ollama.NewPushPullRequest(model), ch)
}

func main() {
	//client := openrouter.NewOpenRouterClient("https://openrouter.ai", os.Getenv("OPENROUTER_KEY")).WithLogFile("openrouter.log")
	//client := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY"))
	client := gemini.NewGeminiClient(os.Getenv("GEMENI_API_KEY"))
	//client := ollama.NewOllamaClient("http://localhost:11434")
	ctx := context.Background()

	registry := tools.NewRegistry()
	registry.Register("run", &RunArgs{}, RunCommand)

	req := ai.NewChatRequest("gemini-3.1-flash-lite-preview").
		WithThinking(true).
		WithTools(registry.GetTools())

	// Initialize the new AgentSession with the request
	agentSession := ai.NewAgentSession(ctx, client, req, ai.WithAccumulator())
	defer agentSession.Stop()

	// Tool executor
	executor := tools.NewToolExecutor(registry)

	// Send the initial user message. This starts the processing loop transparently.
	if err := agentSession.SendUserMessage(ctx, "can you use tool run to get my free disk space?"); err != nil {
		fmt.Printf("\nError: %v\n", err)
	}

	fmt.Println("Streaming Ollama response...")
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
			var resultMsgs []ai.Message
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
			output.WriteString("Thinking:\n")

			output.WriteString(res.ReasoningContent)
			output.WriteString("\n")
		}

		if res.Content != "" {

			output.WriteString(res.Content)
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
