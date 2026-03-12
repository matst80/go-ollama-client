package ai

import (
	"context"
	"encoding/json"
	"testing"
)

type mockChatClient struct {
	ChatClientInterface
	responses []*ChatResponse
}

func (m *mockChatClient) ChatStreamed(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error {

	for _, resp := range m.responses {
		select {
		case ch <- resp:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	close(ch)
	return nil
}

func TestRegistryTools(t *testing.T) {
	ctx := context.Background()
	registry := NewAgentRegistry()
	handler := NewRegistryToolHandler(registry)

	// Register an agent type
	registry.RegisterAgent("echo", AgentDefinition{
		Title:       "Echo Agent",
		Description: "Returns whatever you send",
		SpawnFunction: func(ctx context.Context, content string) AgentSessionInterface {
			client := &mockChatClient{
				responses: []*ChatResponse{
					{
						BaseResponse: &BaseResponse{Done: true},
						Message:      Message{Content: "Echo: " + content},
					},
				},
			}
			req := NewChatRequest("mock")
			return NewAgentSession(ctx, client, req, WithAccumulator())
		},
	})

	// 1. Test list_agent_types
	res, err := handler.Execute(ctx, ToolCall{
		Function: FunctionCall{
			Name: "list_agent_types",
		},
	})
	if err != nil {
		t.Fatalf("list_agent_types failed: %v", err)
	}
	if !contains(res, "Echo Agent") {
		t.Errorf("Expected 'Echo Agent' in list, got %q", res)
	}

	// 2. Test spawn_agent
	args, _ := json.Marshal(map[string]string{
		"type_name":   "echo",
		"instance_id": "echo-1",
		"content":     "Hello world",
	})
	res, err = handler.Execute(ctx, ToolCall{
		Function: FunctionCall{
			Name:      "spawn_agent",
			Arguments: args,
		},
	})
	if err != nil {
		t.Fatalf("spawn_agent failed: %v", err)
	}
	if !contains(res, "spawned successfully") {
		t.Errorf("Expected success message, got %q", res)
	}

	// Verify agent is running
	if _, ok := registry.GetAgent("echo-1"); !ok {
		t.Fatal("Agent 'echo-1' not found in registry")
	}

	// 3. Test list_agents
	res, err = handler.Execute(ctx, ToolCall{
		Function: FunctionCall{
			Name: "list_agents",
		},
	})
	if err != nil {
		t.Fatalf("list_agents failed: %v", err)
	}
	if !contains(res, "echo-1") {
		t.Errorf("Expected 'echo-1' in list, got %q", res)
	}

	// 4. Test message_agent
	args, _ = json.Marshal(map[string]string{
		"instance_id": "echo-1",
		"message":     "Ping",
	})
	res, err = handler.Execute(ctx, ToolCall{
		Function: FunctionCall{
			Name:      "message_agent",
			Arguments: args,
		},
	})
	if err != nil {
		t.Fatalf("message_agent failed: %v", err)
	}
	// Our mock echoes the INITIAL content from SpawnFunction, so it should be "Echo: Hello world"
	if !contains(res, "Echo: Hello world") {
		t.Errorf("Expected response content, got %q", res)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || (len(substr) > 0 && (s[0:len(substr)] == substr || contains(s[1:], substr))))
}

func (m *mockChatClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) { return nil, nil }

