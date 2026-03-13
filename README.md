# go-ollama-client

A unified Go client library for Ollama, Gemini, OpenRouter, and other AI providers. It provides a simple, consistent interface for chat, streaming, and tool execution (function calling).

## Features

- **Unified Interface**: Use the same code to interact with different AI providers.
- **Streaming Support**: Direct support for streaming responses with `ChatStreamed`.
- **Agent Sessions**: High-level `AgentSession` for managing message history and complex interactions.
- **Tool Calling**: Built-in registry and executor for handling model tool calls.
- **Multi-Provider**: Support for Ollama, Gemini, OpenRouter, X.ai, and OpenAI.

## Installation

```bash
go get github.com/matst80/go-ai-agent
```

## Basic Usage

### Simple Chat (Ollama)

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/ollama"
)

func main() {
	client := ollama.NewOllamaClient("http://localhost:11434")
	
	req := ai.NewChatRequest("qwen3.5:4b").
		AddMessage(ai.MessageRoleUser, "Why is the sky blue?")

	resp, err := client.Chat(context.Background(), *req)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Message.Content)
}
```

### Streaming Chat

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/ollama"
)

func main() {
	client := ollama.NewOllamaClient("http://localhost:11434")
	
	req := ai.NewChatRequest("llama3").
		WithStreaming(true).
		AddMessage(ai.MessageRoleUser, "Tell me a story.")

	ch := make(chan *ai.ChatResponse)
	
	go func() {
		err := client.ChatStreamed(context.Background(), *req, ch)
		if err != nil {
			log.Fatal(err)
		}
	}()

	for resp := range ch {
		fmt.Print(resp.Message.Content)
		if resp.Done {
			fmt.Println()
		}
	}
}
```

### Agent Session (High-level API)

The `AgentSession` simplifies handling history and tool execution results.

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/gemini"
)

func main() {
	ctx := context.Background()
	client := gemini.NewGeminiClient(os.Getenv("GEMINI_API_KEY"))
	
	req := ai.NewChatRequest("gemini-3.1-flash-lite-preview").WithStreaming(true)
	
	// Create a session that automatically accumulates history
	session := ai.NewAgentSession(ctx, client, *req, ai.WithAccumulator())
	defer session.Stop()

	// Send a message
	session.SendUserMessage(ctx, "Hello!")

	// Receive as a stream of accumulated responses
	for res := range session.Recv() {
		if res.Chunk.Done {
			fmt.Printf("\nFull Response: %s\n", res.Content)
			break
		}
		fmt.Print(res.Chunk.Message.Content)
	}
}
```

### Using Tools

```go
// Define a tool
type DiskArgs struct {
	Path string `json:"path" tool:"Path to check"`
}

func CheckDisk(args DiskArgs) string {
	return "50GB free"
}

// Register and use
registry := tools.NewRegistry()
registry.Register("check_disk", &DiskArgs{}, CheckDisk)

req := ai.NewChatRequest("model").WithTools(registry.GetTools())
```

## Registry Tools & Multi-Agent Orchestration

The `AgentRegistry` and `RegistryToolHandler` enable complex multi-agent workflows where a "master" agent can spawn, list, and message other sub-agents.

### 1. Register Agent Types
Define the types of agents that can be dynamically spawned.

```go
registry := ai.NewAgentRegistry()

registry.RegisterAgent("ollama", ai.AgentDefinition{
    Title:       "Ollama Agent",
    Description: "Local LLM powered by Ollama",
    SpawnFunction: func(ctx context.Context, content string) ai.AgentSessionInterface {
        client := ollama.NewOllamaClient("http://localhost:11434")
        req := ai.NewChatRequest("llama3")
        req.Messages = []ai.Message{{Role: ai.MessageRoleSystem, Content: content}}
        return ai.NewAgentSession(ctx, client, req)
    },
})
```

### 2. Expose Registry as Tools
Use `RegistryToolHandler` to convert registry operations into tools that an LLM can understand and call.

```go
// Create the handler
toolHandler := ai.NewRegistryToolHandler(registry)

// Register these tools with a ToolRegistry
masterToolRegistry := tools.NewRegistry()
masterToolRegistry.RegisterTools(toolHandler.GetToolDefinitions()...)

// Create a Master Agent that has access to these tools
masterClient := gemini.NewGeminiClient(apiKey)
masterReq := ai.NewChatRequest("gemini-1.5-flash").WithTools(masterToolRegistry.GetTools())
masterSession := ai.NewAgentSession(ctx, masterClient, masterReq)
```

### Available Registry Tools

- `spawn_agent`: Spawns a new instance of a registered agent type (e.g., "researcher").
- `message_agent`: Sends a message to a running agent instance and returns its response.
- `list_agents`: Returns a list of all currently active agent instances.
- `list_agent_types`: Returns all available agent types that can be spawned.

## Supported Providers

- **Ollama**: `ollama.NewOllamaClient(url)`
- **Gemini**: `gemini.NewGeminiClient(apiKey)`
- **OpenRouter**: `openrouter.NewOpenRouterClient(url, apiKey)`
- **OpenAI**: `openai.NewOpenAIClient(url, apiKey)`
- **X.ai**: `xai.NewXAIClient(url, apiKey)`

## License

MIT
