package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// RegistryToolHandler provides tools for agents to interact with the AgentRegistry
type RegistryToolHandler struct {
	Registry *AgentRegistry
}

func NewRegistryToolHandler(registry *AgentRegistry) *RegistryToolHandler {
	return &RegistryToolHandler{Registry: registry}
}

func (h *RegistryToolHandler) GetTools() []Tool {
	return []Tool{
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "spawn_agent",
				Description: "Spawn a new agent instance of a given type. Use unique instance_id.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"type_name":   map[string]any{"type": "string", "description": "The type of agent to spawn (e.g. 'researcher', 'critic')"},
						"instance_id": map[string]any{"type": "string", "description": "A unique ID for this instance"},
						"content":     map[string]any{"type": "string", "description": "Initial instructions or message for the new agent"},
					},
					"required": []string{"type_name", "instance_id", "content"},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "message_agent",
				Description: "Send a message to a running agent instance and wait for response.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"instance_id": map[string]any{"type": "string", "description": "The ID of the running agent instance"},
						"message":     map[string]any{"type": "string", "description": "The message to send"},
					},
					"required": []string{"instance_id", "message"},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "list_agents",
				Description: "List all currently running agent instances.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "list_agent_types",
				Description: "List all available agent types that can be spawned.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}
}

func (h *RegistryToolHandler) Execute(ctx context.Context, call ToolCall) (string, error) {
	switch call.Function.Name {
	case "spawn_agent":
		var args struct {
			TypeName   string `json:"type_name"`
			InstanceID string `json:"instance_id"`
			Content    string `json:"content"`
		}
		if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
			return "", err
		}
		_, err := h.Registry.SpawnAgent(ctx, args.TypeName, args.InstanceID, args.Content)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Agent instance '%s' spawned successfully.", args.InstanceID), nil

	case "message_agent":
		var args struct {
			InstanceID string `json:"instance_id"`
			Message    string `json:"message"`
		}
		if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
			return "", err
		}

		agent, ok := h.Registry.GetAgent(args.InstanceID)
		if !ok {
			return "", fmt.Errorf("agent instance '%s' not found", args.InstanceID)
		}

		// Send the message
		if err := agent.SendUserMessage(ctx, args.Message); err != nil {
			return "", err
		}

		// Wait for response and accumulate it
		// This assumes the agent is configured to emit something useful, 
		// but since we only have SendUserMessage, we need to read from GlobalChan.
		// For registry communication, we probably want to wait until 'done'.
		
		var lastContent string
		recv := agent.RecvAny()
		for {

			select {
			case res, ok := <-recv:
				if !ok {
					return lastContent, nil
				}
				// We need to handle the type of res.
				// If it's *AccumulatedResponse, we take the content.
				// If it's *ChatResponse, it's streaming, so we might need to accumulate it here.
				
				switch v := any(res).(type) {
				case *AccumulatedResponse:
					if v.Chunk != nil && v.Chunk.Done {
						return v.Content, nil
					}
					lastContent = v.Content
				case *ChatResponse:
					// This is problematic if it's just chunks.
					// We'd need to accumulate it.
					if v.Done {
						return "Message sent, but response accumulation not supported for raw ChatResponse", nil
					}
				default:
					// Generic fallback
					lastContent = fmt.Sprintf("%v", v)
				}
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}

	case "list_agents":
		agents := h.Registry.GetRunningAgents()
		if len(agents) == 0 {
			return "No agents currently running.", nil
		}
		var sb strings.Builder
		sb.WriteString("Running agents:\n")
		for id := range agents {
			sb.WriteString(fmt.Sprintf("- %s\n", id))
		}
		return sb.String(), nil

	case "list_agent_types":
		h.Registry.mu.RLock()
		defer h.Registry.mu.RUnlock()
		if len(h.Registry.AgentTypes) == 0 {
			return "No agent types registered.", nil
		}
		var sb strings.Builder
		sb.WriteString("Available agent types:\n")
		for name, def := range h.Registry.AgentTypes {
			sb.WriteString(fmt.Sprintf("- %s: %s (%s)\n", name, def.Title, def.Description))
		}
		return sb.String(), nil

	default:
		return "", fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}
