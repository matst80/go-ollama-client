package tools

import (
	"context"
	"encoding/json"
	"testing"
)

type TestArgs struct {
	Command              string `json:"command" tool:"Command to run,required"`
	CommandLineArguments string `json:"arg" tool:"Arguments to the command"`
	Wait                 bool   `json:"wait" tool:"Wait for command to finish"`
}

func TestRegistry_GetTools(t *testing.T) {
	registry := NewRegistry()

	handler := func(args TestArgs) {}
	registry.Register("run", "run a command", &TestArgs{}, handler)

	tools := registry.GetTools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	tool := tools[0]
	if tool.Function.Name != "run" {
		t.Errorf("expected tool name 'run', got '%s'", tool.Function.Name)
	}

	params := tool.Function.Parameters.(map[string]interface{})
	if params["type"] != "object" {
		t.Errorf("expected type 'object', got '%s'", params["type"])
	}

	properties := params["properties"].(map[string]interface{})
	if _, ok := properties["command"]; !ok {
		t.Error("expected property 'command' not found")
	}

	commandProp := properties["command"].(map[string]interface{})
	if commandProp["type"] != "string" {
		t.Errorf("expected command type 'string', got '%s'", commandProp["type"])
	}
	if commandProp["description"] != "Command to run" {
		t.Errorf("expected command description 'Command to run', got '%s'", commandProp["description"])
	}

	required := params["required"].([]string)
	foundRequired := false
	for _, r := range required {
		if r == "command" {
			foundRequired = true
			break
		}
	}
	if !foundRequired {
		t.Error("expected 'command' to be in required list")
	}
}

func TestRegistry_Call(t *testing.T) {
	registry := NewRegistry()

	var calledWith TestArgs
	handler := func(args TestArgs) string {
		calledWith = args
		return "ok"
	}

	registry.Register("run", "run a command", &TestArgs{}, handler)

	argsJSON := json.RawMessage(`{"command": "ls", "arg": "-la", "wait": true}`)
	results, err := registry.Call(context.Background(), "run", argsJSON)
	if err != nil {
		t.Fatalf("failed to call tool: %v", err)
	}

	if calledWith.Command != "ls" {
		t.Errorf("expected command 'ls', got '%s'", calledWith.Command)
	}
	if calledWith.CommandLineArguments != "-la" {
		t.Errorf("expected arg '-la', got '%s'", calledWith.CommandLineArguments)
	}
	if !calledWith.Wait {
		t.Error("expected wait to be true")
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 return value, got %d", len(results))
	}
	if results[0].String() != "ok" {
		t.Errorf("expected return 'ok', got '%s'", results[0].String())
	}
}
