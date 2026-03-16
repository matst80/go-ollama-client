package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/ollama"
	"github.com/matst80/go-ai-agent/pkg/openrouter"
	"github.com/matst80/go-ai-agent/pkg/tools"
	"github.com/matst80/go-ai-agent/pkg/xai"
)

func main() {
	ctx := context.Background()

	// Enable a package-level default logfile for all ApiClient instances if the environment
	// variable `AI_LOG_PATH` is set. This is picked up by NewApiClient so underlying clients
	// (Ollama, OpenRouter, xAI, etc.) will inherit logging automatically.
	if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
		ai.SetDefaultLogFile(lp)
	}

	// 1. Initialize AgentRegistry
	registry := ai.NewAgentRegistry()

	// 2. Register Agent Types: Ollama
	registry.RegisterAgent("ollama", ai.NewAgentDefinition(
		"Ollama Agent",
		"Local LLM powered by Ollama (llama3)",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := ollama.NewOllamaClient("http://localhost:11434")
			req := ai.NewChatRequest("qwen3.5:4b")
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))

	// 3. Register Agent Types: xAI
	registry.RegisterAgent("xai", ai.NewAgentDefinition(
		"xAI Agent",
		"Cloud LLM powered by xAI (grok-beta)",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY"))
			// If a global log path is configured, make sure the client also has its local log path set.
			if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
				client.WithLogFile(lp)
			}
			req := ai.NewChatRequest("grok-beta")
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))
	registry.RegisterAgent("openrouter", ai.NewAgentDefinition(
		"OpenRouter Agent",
		"Cloud LLM powered by OpenRouter (hunter-alpha)",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := openrouter.NewOpenRouterClient("https://openrouter.ai", os.Getenv("OPENROUTER_API_KEY"))
			// Ensure the OpenRouter client also gets its logfile set if provided.
			if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
				client.WithLogFile(lp)
			}
			req := ai.NewChatRequest("openrouter/hunter-alpha")
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))

	// 4. Create RegistryToolHandler to expose registry operations as tools
	toolHandler := ai.NewRegistryToolHandler(registry)

	// 5. Setup Tool Registry for the "Master" agent
	masterToolRegistry := tools.NewRegistry()
	masterToolRegistry.RegisterTools(toolHandler.GetToolDefinitions()...)

	// 6. Setup Master Agent (using xAI here, but could be Gemini/Ollama)
	masterClient := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY"))
	// Ensure master client logging is enabled locally as well when configured.
	if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
		masterClient.WithLogFile(lp)
	}
	// System prompt: instruct the model how to emit streamed file operations.
	// Prefer the fenced `diffstream` blocks to avoid JSON escaping, but NDJSON
	// is still supported. The system will accept either format and will process
	// the operations into the repository workspace.
	// Examples (fenced):
	// ```diffstream type=file op=add path=workspace/info.txt encoding=utf-8
	// The file contents here
	// ```
	// Chunked (binary) example:
	// ```diffstream type=chunk file_id=f1 chunk_index=0 total_chunks=2 data_encoding=base64
	// BASE64DATA
	// ```
	// NDJSON example (still accepted):
	// {"type":"file","op":"add","path":"src/foo.go","content_encoding":"utf-8","content":"package main\n\nfunc Foo() {}\n","atomic":true}
	// When operations are processed the agent will receive a [diff-report] summary
	// listing OK/FAILED operations. If a change cannot be made, emit a
	// type=meta message with an explanatory `message` field.

	systemPrompt := "Output machine-actionable file changes using fenced `diffstream` blocks only. emit only fenced blocks when performing edits.\n" +
		"Examples (file add):\n" +
		"```diffstream type=file op=add path=workspace/info.txt encoding=utf-8\nThe single-line content goes here.\n```\n" +
		"Chunked text upload example (large text split across multiple chunk fences):\n" +
		"```diffstream type=chunk file_id=f1 chunk_index=0 total_chunks=2 data_encoding=utf-8\nFirst part of the file...\n```\n" +
		"```diffstream type=chunk file_id=f1 chunk_index=1 total_chunks=2 data_encoding=utf-8\nSecond part of the file...\n```\n" +
		"Commit example:\n" +
		"```diffstream type=commit message=\"Add info.txt\" finalize=true\n\n```\n" +
		"Note: binary files can still be uploaded using base64-encoded chunk fences, but avoid emitting base64 in examples. After processing, the system will emit a [diff-report] summary showing which operations succeeded or failed.\n"

	masterReq := ai.NewChatRequest("grok-4-1-fast-non-reasoning").WithTools(masterToolRegistry.GetTools())
	// place the system prompt as the first message
	masterReq.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: systemPrompt}}

	// Create the master session and wire the diff parser into it via options.
	repoRoot := "./test-repo"
	_ = os.MkdirAll(repoRoot, 0o755)

	masterSession := ai.NewAgentSession(ctx, masterClient, masterReq, ai.WithRepoRoot(repoRoot), ai.WithOperationHandler(&ai.DefaultOperationHandler{}))
	defer masterSession.Stop()
	executor := tools.NewToolExecutor(masterToolRegistry)

	// 7. Simple Test: Ask the Master Agent to spawn an OpenRouter agent and talk to it
	fmt.Println("--- Master Agent Session Started ---")
	testPrompt := "Use fenced `diffstream` blocks to create a file at 'workspace/sky.md' containing the answer to why the sky is blue. After applying the change, send a short confirmation message."

	if err := masterSession.SendUserMessage(ctx, testPrompt); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	var lastLines []string
	for res := range masterSession.Recv() {
		// Handle Tool Calls
		if res.Chunk.Done && len(res.ToolCalls) > 0 {
			results, err := executor.HandleCalls(ctx, res.ToolCalls)
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
	fmt.Printf("Final list of agents:\n")
	for id, agent := range toolHandler.Registry.GetRunningAgents() {
		fmt.Printf("Agent %s history:\n", id)
		for _, msg := range agent.GetMessageHistory() {
			fmt.Printf("  %s: %s\n", msg.Role, msg.Content)
		}
	}
}
