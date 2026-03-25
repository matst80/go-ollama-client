package mcp

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/tools"
)

func TestServerProxy(t *testing.T) {
	s := server.NewMCPServer("test-server", "1.0.0")

	// Add a dummy tool
	tool := mcp.NewTool("echo",
		mcp.WithDescription("Echos the input text"),
	)

	s.AddTool(tool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		args, ok := req.Params.Arguments.(map[string]interface{})
		if !ok {
			return mcp.NewToolResultError("invalid arguments"), nil
		}
		message := args["message"].(string)
		return mcp.NewToolResultText(message), nil
	})

	inProcessClient, err := client.NewInProcessClient(s)
	if err != nil {
		t.Fatalf("failed to create in-process client: %v", err)
	}

	proxy := NewServerProxy(inProcessClient)
	ctx := context.Background()

	err = proxy.Initialize(ctx)
	if err != nil {
		t.Fatalf("failed to initialize proxy: %v", err)
	}

	registry := tools.NewRegistry()
	err = proxy.RegisterTools(ctx, registry)
	if err != nil {
		t.Fatalf("failed to register tools: %v", err)
	}

	if !registry.HasTool("echo") {
		t.Fatalf("registry is missing 'echo' tool")
	}

	args := map[string]interface{}{"message": "hello world"}
	argsBytes, _ := json.Marshal(args)

	result, err := registry.Call(ctx, "echo", argsBytes)
	if err != nil {
		t.Fatalf("failed to call 'echo' tool: %v", err)
	}

	res, ok := result[0].Interface().(ai.MultimodalToolResult)
	if !ok {
		t.Fatalf("expected result type ai.MultimodalToolResult, got %T", result[0].Interface())
	}
	if res.Content != "hello world" {
		t.Errorf("expected 'hello world', got '%s'", res.Content)
	}
}
