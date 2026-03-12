package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/matst80/go-ai-agent/pkg/ai"
	"github.com/matst80/go-ai-agent/pkg/tools"
)

// ServerProxy manages an MCP client and registers its tools with the agent registry.
type ServerProxy struct {
	client client.MCPClient
}

// NewServerProxy creates a new ServerProxy wrapping the given MCP client.
func NewServerProxy(mcpClient client.MCPClient) *ServerProxy {
	return &ServerProxy{
		client: mcpClient,
	}
}

// NewStdioServerProxy creates and starts an MCP client over standard I/O and returns a proxy.
func NewStdioServerProxy(command string, env []string, args ...string) (*ServerProxy, error) {
	c, err := client.NewStdioMCPClient(command, env, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to create stdio MCP client: %w", err)
	}
	return NewServerProxy(c), nil
}

// NewSSEServerProxy creates and starts an MCP client over SSE and returns a proxy.
func NewSSEServerProxy(baseURL string) (*ServerProxy, error) {
	c, err := client.NewSSEMCPClient(baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create SSE MCP client: %w", err)
	}
	return NewServerProxy(c), nil
}

// Initialize sends the initial connection request to the server. Must be called before RegisterTools.
func (p *ServerProxy) Initialize(ctx context.Context) error {
	req := mcp.InitializeRequest{}
	req.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	req.Params.ClientInfo = mcp.Implementation{
		Name:    "go-ai-agent-proxy",
		Version: "1.0.0",
	}

	_, err := p.client.Initialize(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to initialize MCP client: %w", err)
	}
	return nil
}

// RegisterTools fetches all tools from the connected MCP server and registers them with the provided registry.
func (p *ServerProxy) RegisterTools(ctx context.Context, registry *tools.Registry) error {
	return p.RegisterToolsWithFilter(ctx, registry, nil)
}

// RegisterToolsWithFilter fetches all tools from the connected MCP server and registers them, 
// skipping any tools whose names exist in the disabledTools map.
func (p *ServerProxy) RegisterToolsWithFilter(ctx context.Context, registry *tools.Registry, disabledTools map[string]bool) error {
	res, err := p.client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	for _, t := range res.Tools {
		// Skip disabled tools
		if disabledTools != nil && disabledTools[t.Name] {
			continue
		}

		// Calculate parameters map
		var params map[string]interface{}
		
		var schemaData []byte
		if t.InputSchema.Type != "" {
			schemaData, _ = json.Marshal(t.InputSchema)
		} else if len(t.RawInputSchema) > 0 {
			schemaData = t.RawInputSchema
		} else {
			// default empty object schema
			schemaData = []byte(`{"type":"object","properties":{}}`)
		}

		if err := json.Unmarshal(schemaData, &params); err != nil {
			return fmt.Errorf("failed to unmarshal JSON schema for tool %s: %w", t.Name, err)
		}

		argsType := reflect.TypeOf(map[string]interface{}{})
		toolName := t.Name

		handlerFunc := func(args map[string]interface{}) string {
			callReq := mcp.CallToolRequest{}
			callReq.Params.Name = toolName
			callReq.Params.Arguments = args

			callRes, err := p.client.CallTool(context.Background(), callReq)
			if err != nil {
				return fmt.Sprintf("failed to call tool %s: %v", toolName, err)
			}
			if callRes.IsError {
				return fmt.Sprintf("tool %s returned error: %v", toolName, mcp.GetTextFromContent(callRes.Content))
			}

			var out string
			for _, c := range callRes.Content {
				out += mcp.GetTextFromContent(c)
			}
			return out
		}

		def := ai.ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			Enabled:     true,
			ArgsType:    argsType,
			Handler:     reflect.ValueOf(handlerFunc),
			Parameters:  params,
		}

		if err := registry.RegisterTool(def); err != nil {
			return fmt.Errorf("failed to register tool %s: %w", t.Name, err)
		}
	}

	return nil
}

// Close closes the underlying MCP client connection.
func (p *ServerProxy) Close() error {
	return p.client.Close()
}
