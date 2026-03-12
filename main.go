package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/ollama"
	"github.com/matst80/go-ai-agent/pkg/tools"
	"github.com/matst80/go-ai-agent/pkg/xai"
)

func main() {
	ctx := context.Background()

	// 1. Initialize AgentRegistry
	registry := ai.NewAgentRegistry()

	// 2. Register Agent Types: Ollama
	registry.RegisterAgent("ollama", ai.AgentDefinition{
		Title:       "Ollama Agent",
		Description: "Local LLM powered by Ollama (llama3)",
		SpawnFunction: func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := ollama.NewOllamaClient("http://localhost:11434")
			req := ai.NewChatRequest("qwen3.5:4b")
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession[*ai.AccumulatedResponse](ctx, client, req, ai.WithAccumulator())
		},
	})

	// 3. Register Agent Types: xAI
	registry.RegisterAgent("xai", ai.AgentDefinition{
		Title:       "xAI Agent",
		Description: "Cloud LLM powered by xAI (grok-beta)",
		SpawnFunction: func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY"))
			req := ai.NewChatRequest("grok-beta")
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession[*ai.AccumulatedResponse](ctx, client, req, ai.WithAccumulator())
		},
	})

	// 4. Create RegistryToolHandler to expose registry operations as tools
	toolHandler := ai.NewRegistryToolHandler(registry)

	// 5. Setup Tool Registry for the "Master" agent
	masterToolRegistry := tools.NewRegistry()
	masterToolRegistry.RegisterTools(toolHandler.GetToolDefinitions()...)

	// 6. Setup Master Agent (using xAI here, but could be Gemini/Ollama)
	masterClient := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY"))
	masterReq := ai.NewChatRequest("grok-4-1-fast-non-reasoning").
		WithTools(masterToolRegistry.GetTools())

	masterSession := ai.NewAgentSession[*ai.AccumulatedResponse](ctx, masterClient, masterReq, ai.WithAccumulator())
	defer masterSession.Stop()

	executor := tools.NewToolExecutor(masterToolRegistry)

	// 7. Simple Test: Ask the Master Agent to spawn an Ollama agent and talk to it
	fmt.Println("--- Master Agent Session Started ---")
	testPrompt := "Please spawn an 'ollama' agent with instance ID 'local-assistant' to help with local tasks. " +
		"Then message it to ask 'What is the capital of France?' and report back."

	if err := masterSession.SendUserMessage(ctx, testPrompt); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	var lastLines []string
	for res := range masterSession.Recv() {
		// Handle Tool Calls
		if res.Chunk.Done && len(res.ToolCalls) > 0 {
			results, err := executor.HandleCalls(res.ToolCalls)
			if err != nil {
				fmt.Printf("Tool execution error: %v\n", err)
			}

			var resultMsgs []ai.Message
			for _, tr := range results {
				msg := tr.ToResultMessage()
				resultMsgs = append(resultMsgs, *msg)
				fmt.Printf("\n[tool result] %s\n", msg.Content)
			}

			if len(resultMsgs) > 0 {
				if err := masterSession.SendMessages(ctx, resultMsgs...); err != nil {
					fmt.Printf("failed to deliver tool results: %v\n", err)
				}
			}
		}

		// UI Output (live update)
		if res.Content != "" {
			outStr := res.Content
			lines := strings.Split(strings.TrimRight(outStr, "\n"), "\n")

			diffLine := 0
			for diffLine < len(lines) && diffLine < len(lastLines) && lines[diffLine] == lastLines[diffLine] {
				diffLine++
			}

			if diffLine == len(lines) && len(lines) == len(lastLines) {
				if res.Chunk.Done && len(res.ToolCalls) == 0 {
					break
				}
				continue
			}

			if len(lastLines) > 0 {
				moveUp := len(lastLines) - diffLine
				if moveUp > 0 {
					fmt.Printf("\033[%dA\r\033[J", moveUp)
				}
			}

			for i := diffLine; i < len(lines); i++ {
				fmt.Println(lines[i])
			}
			lastLines = lines
		}

		if res.Chunk.Done && len(res.ToolCalls) == 0 {
			break
		}
	}

	fmt.Println("\n--- Test Completed ---")
	fmt.Printf("Final list of agents:\n%s\n", toolHandler.Registry.GetRunningAgents())
}
