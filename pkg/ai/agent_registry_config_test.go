package ai

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

func TestAgentRegistryConfig(t *testing.T) {
	ctx := context.Background()
	config := NewAgentRegistryConfig()

	// 1. Register a mock client
	mockClient := &mockChatClient{}
	config.RegisterClient("mock", mockClient)

	// // 2. Register a tool
	// type TestArgs struct {
	// 	Value string `json:"value"`
	// }
	// toolDef, _ := GetToolDefinition("test_tool", "A test tool", TestArgs{}, func(args TestArgs) string {
	// 	return "Result: " + args.Value
	// })
	//config.WithTools(*toolDef)

	// 3. Register an agent via config
	agentConf := AgentConfig{
		Name:         "test-agent",
		Title:        "Test Agent",
		Description:  "An agent for testing config",
		SystemPrompt: "You are a test-agent.",
		Model:        "gpt-test",
		Client:       "mock",
	}
	config.RegisterAgentConfig(agentConf)

	// 4. Add OnSpawn hook
	spawnCalled := false
	config.OnSpawn = func(ctx context.Context, name string, session AgentSessionInterface) {
		if name == "test-agent" {
			spawnCalled = true
		}
	}

	// 5. Build the registry
	registry := config.Build()

	// 6. Spawn the agent
	session, err := registry.SpawnAgent(ctx, "test-agent", "instance-1", "Initial message", NewDefaultAgentState())
	if err != nil {
		t.Fatalf("Failed to spawn agent: %v", err)
	}

	if !spawnCalled {
		t.Error("OnSpawn hook was not called")
	}

	// 7. Verify session state/config
	state := session.GetState()
	if state.GetTitle() != "Test Agent" {
		t.Errorf("Expected title 'Test Agent', got %q", state.GetTitle())
	}

	// Since we can't easily inspect tools on the session without more interfaces,
	// we'll check if the session is correctly initialized.
	// We can cast to *AgentSession to check internal state for testing.
	s, ok := session.(*AgentSession)
	if !ok {
		t.Fatal("session is not *AgentSession")
	}

	if s.rec.Model != "gpt-test" {
		t.Errorf("Expected model 'gpt-test', got %q", s.rec.Model)
	}

	// if len(s.rec.Tools) != 1 || s.rec.Tools[0].Function.Name != "test_tool" {
	// 	t.Errorf("Expected 1 tool 'test_tool', got %+v", s.rec.Tools)
	// }

	if len(s.rec.Messages) < 2 {
		t.Fatalf("Expected at least 2 messages (system + initial), got %d", len(s.rec.Messages))
	}
	if s.rec.Messages[0].Content != "You are a test-agent." {
		t.Errorf("Expected system prompt, got %q", s.rec.Messages[0].Content)
	}
}

func TestLoadAgentsFromFile(t *testing.T) {
	tmpFile := "test_agents.json"
	agents := map[string]AgentConfig{
		"file-agent": {
			Name:        "file-agent",
			Title:       "File Agent",
			Description: "Loaded from file",
			Client:      "mock",
		},
	}
	data, _ := json.Marshal(agents)
	_ = os.WriteFile(tmpFile, data, 0644)
	defer os.Remove(tmpFile)

	config := NewAgentRegistryConfig()
	err := config.LoadAgentsFromFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load agents: %v", err)
	}

	if _, ok := config.Agents["file-agent"]; !ok {
		t.Error("Agent 'file-agent' not found in config")
	}
}

func TestOnChatRequestHook(t *testing.T) {
	ctx := context.Background()
	mockClient := &mockChatClient{}

	hookCalled := false
	session := NewAgentSession(ctx, mockClient, NewChatRequest("mock"), NewDefaultAgentState(),
		WithOnChatRequest(func(ctx context.Context, req *ChatRequest) error {
			hookCalled = true
			req.Model = "hooked-model"
			return nil
		}))

	// Trigger a chat
	_ = session.SendUserMessage(ctx, "Hello")

	if !hookCalled {
		t.Error("OnChatRequest hook was not called")
	}

	if session.rec.Model != "hooked-model" {
		t.Error("OnChatRequest hook failed to modify request")
	}
}
