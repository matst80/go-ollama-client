package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/github"
	"github.com/matst80/go-ai-agent/pkg/mcp"
	"github.com/matst80/go-ai-agent/pkg/ollama"
	"github.com/matst80/go-ai-agent/pkg/openrouter"
	"github.com/matst80/go-ai-agent/pkg/terminal"
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
			client := ollama.NewOllamaClient("http://localhost:11434").WithDefaultModel("qwen3.5:4b")
			req := ai.NewDefaultChatRequest()
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))

	// 3. Register Agent Types: xAI
	registry.RegisterAgent("xai", ai.NewAgentDefinition(
		"xAI Agent",
		"Cloud LLM powered by xAI (grok-beta)",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := xai.NewXAIClient("https://api.x.ai/v1", os.Getenv("XAI_API_KEY")).WithDefaultModel("grok-beta")
			// If a global log path is configured, make sure the client also has its local log path set.
			if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
				client.WithLogFile(lp)
			}
			req := ai.NewDefaultChatRequest()
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))
	registry.RegisterAgent("openrouter", ai.NewAgentDefinition(
		"OpenRouter Agent",
		"Cloud LLM powered by OpenRouter (hunter-alpha)",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			client := openrouter.NewOpenRouterClient("https://openrouter.ai", os.Getenv("OPENROUTER_API_KEY")).WithDefaultModel("openrouter/hunter-alpha")
			// Ensure the OpenRouter client also gets its logfile set if provided.
			if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
				client.WithLogFile(lp)
			}
			req := ai.NewDefaultChatRequest()
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))
	registry.RegisterAgent("github", ai.NewAgentDefinition(
		"GitHub Agent",
		"Cloud LLM powered by GitHub Models",
		func(ctx context.Context, content string) ai.AgentSessionInterface {
			model := os.Getenv("GITHUB_MODEL")
			if model == "" {
				model = "gpt-4o"
			}
			client := github.NewGitHubClient(os.Getenv("GITHUB_TOKEN"), "").WithDefaultModel(model)
			if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
				client.WithLogFile(lp)
			}
			req := ai.NewDefaultChatRequest()
			req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
			return ai.NewAgentSession(ctx, client, req)
		},
	))

	// 4. Create RegistryToolHandler to expose registry operations as tools
	toolHandler := ai.NewRegistryToolHandler(registry)

	masterToolRegistry := tools.NewRegistry()
	masterToolRegistry.RegisterTools(toolHandler.GetToolDefinitions()...)

	// 6. Setup MCP Servers
	var mcpProxies []*mcp.ServerProxy
	if _, err := os.Stat("mcp_config.json"); err == nil {
		fmt.Println("Loading MCP servers from mcp_config.json...")
		proxies, err := mcp.LoadAndRegisterServers(ctx, "mcp_config.json", masterToolRegistry)
		if err != nil {
			fmt.Printf("Warning: failed to load MCP servers: %v\n", err)
		} else {
			mcpProxies = proxies
			fmt.Printf("Registered tools from %d MCP servers\n", len(mcpProxies))
			// Print tool names for debugging
			for _, tool := range masterToolRegistry.GetTools() {
				fmt.Printf("Tool available: %s\n", tool.Function.Name)
			}
		}
	}
	defer func() {
		for _, p := range mcpProxies {
			p.Close()
		}
	}()

	// Setup Master Agent (using GitHub for verification)
	masterClient := github.NewGitHubClient(os.Getenv("GITHUB_TOKEN"), "").WithDefaultModel("gpt-4o")
	if lp := os.Getenv("AI_LOG_PATH"); lp != "" {
		masterClient.WithLogFile(lp)
	}

	systemPrompt := "Output machine-actionable file changes using fenced `diff` blocks only. Do not emit surrounding prose when performing edits.\n" +
		"Each fenced block must contain an exact unified git diff that can be applied directly with git apply.\n" +
		"Example:\n" +
		"```diff\n--- a/main.go\n+++ b/main.go\n@@ -12,5 +12,5 @@ func add(a int, b int) int {\n }\n+// Computes the sum of two integer arguments\n-// add is a simple function that returns the sum of two integers\n func main() {\n```\n" +
		"For new files, use standard git diff format such as --- /dev/null and +++ b/path/to/file.\n" +
		"After processing, the system will emit a [diff-report] summary showing which operations succeeded or failed.\n"

	masterReq := ai.NewDefaultChatRequest().WithTools(masterToolRegistry.GetTools())
	// place the system prompt as the first message
	masterReq.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: systemPrompt}}

	// Create the master session and wire the diff parser into it via options.
	repoRoot := "./test-repo"
	_ = os.MkdirAll(repoRoot, 0o755)

	// Create a SummarizeTruncator that uses the xAI client to summarize conversation
	summarizer := ai.NewSummarizeTruncator(masterClient, &ai.SummarizeOptions{
		Threshold:              50,
		RemoveCount:            10,
		Model:                  masterReq.Model,
		SummaryPrompt:          "Summarize the following conversation into concise bullets with any action items.",
		Timeout:                10 * time.Second,
		TokenEstimateThreshold: 20000,
		Logger:                 ai.NoopLogger{},
	})

	executor := tools.NewToolExecutor(masterToolRegistry)
	masterSession := ai.NewAgentSession(ctx, masterClient, masterReq,
		ai.WithRepoRoot(repoRoot),
		ai.WithOperationHandler(&ai.DefaultOperationHandler{}),
		ai.WithTruncation(summarizer),
		executor.AgentSessionOption(func(tr tools.ToolResult) {
			if tr.Err != nil {
				fmt.Printf("\n[tool error] %s\n", tr.Err)
			} else {
				fmt.Printf("\n[tool result] %s\n", tr.Content)
			}
		}),
	)
	defer masterSession.Stop()

	// 8. Simple Test: Ask the Master Agent to navigate with browser and screenshot
	fmt.Println("--- Master Agent Session Started ---")
	testPrompt := "List available tools. Then, use a browser tool to navigate to google.com. Does it contain a search field?"

	if err := masterSession.SendUserMessage(ctx, testPrompt); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	updater := terminal.NewTTYUpdater()
	for res := range masterSession.Recv() {
		if updater.Handle(res) {
			break
		}
	}

	fmt.Println("\n--- First message Completed ---")
	fmt.Printf("List of agents:\n")
	for id, agent := range toolHandler.Registry.GetRunningAgents() {
		fmt.Printf("Agent %s history:\n", id)
		for _, msg := range agent.GetMessageHistory() {
			fmt.Printf("  %s: %s\n", msg.Role, msg.Content)
		}
	}

	// 2. Ask the Master Agent to spawn an OpenRouter agent and talk to it
	testPrompt2 := "Can you enter 'slask' in the search field on the current page in the browser?"
	if err := masterSession.SendUserMessage(ctx, testPrompt2); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	for res := range masterSession.Recv() {
		if updater.Handle(res) {
			break
		}
	}

	fmt.Println("\n--- Test Completed ---")
	fmt.Printf("List of agents:\n")
	for id, agent := range toolHandler.Registry.GetRunningAgents() {
		fmt.Printf("Agent %s history:\n", id)
		for _, msg := range agent.GetMessageHistory() {
			fmt.Printf("  %s: %s\n", msg.Role, msg.Content)
		}
	}
}
