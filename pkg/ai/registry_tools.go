package ai

import (
	"context"
	"fmt"
	"strings"
)

// RegistryToolHandler provides tools for agents to interact with the AgentRegistry
type RegistryToolHandler struct {
	Registry *AgentRegistry
	tools    map[string]ToolDefinition
}

func NewRegistryToolHandler(registry *AgentRegistry) *RegistryToolHandler {
	h := &RegistryToolHandler{
		Registry: registry,
		tools:    make(map[string]ToolDefinition),
	}
	h.initTools()
	return h
}

type SpawnAgentArgs struct {
	TypeName   string `json:"type_name" tool:"The type of agent to spawn (e.g. 'researcher', 'critic'),required"`
	InstanceID string `json:"instance_id" tool:"A unique ID for this instance,required"`
	Content    string `json:"content" tool:"Initial instructions or message for the new agent,required"`
}

type MessageAgentArgs struct {
	InstanceID string `json:"instance_id" tool:"The ID of the running agent instance,required"`
	Message    string `json:"message" tool:"The message to send,required"`
}

type ListAgentsArgs struct{}

type ListAgentTypesArgs struct{}

func (h *RegistryToolHandler) initTools() {
	tools := []struct {
		name string
		desc string
		args any
		fn   any
	}{
		{"spawn_agent", "Spawn a new agent instance of a given type. Use unique instance_id.", SpawnAgentArgs{}, h.spawnAgent},
		{"message_agent", "Send a message to a running agent instance and wait for response.", MessageAgentArgs{}, h.messageAgent},
		{"list_agents", "List all currently running agent instances.", ListAgentsArgs{}, h.listAgents},
		{"list_agent_types", "List all available agent types that can be spawned.", ListAgentTypesArgs{}, h.listAgentTypes},
	}

	for _, t := range tools {
		def, err := GetToolDefinition(t.name, t.desc, t.args, t.fn)
		if err == nil {
			h.tools[t.name] = *def
		}
	}
}

func (h *RegistryToolHandler) GetToolDefinitions() []ToolDefinition {
	var tools []ToolDefinition
	for _, def := range h.tools {
		tools = append(tools, def)
	}
	return tools
}

func (h *RegistryToolHandler) GetTools() []Tool {
	var tools []Tool
	for _, def := range h.tools {
		tools = append(tools, Tool{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Parameters,
			},
		})
	}
	return tools
}

func (h *RegistryToolHandler) spawnAgent(args SpawnAgentArgs) string {
	_, err := h.Registry.SpawnAgent(context.Background(), args.TypeName, args.InstanceID, args.Content, "master")
	if err != nil {
		return fmt.Sprintf("failed to spawn agent: %v", err)
	}
	return fmt.Sprintf("Agent instance '%s' spawned successfully.", args.InstanceID)
}

func (h *RegistryToolHandler) messageAgent(args MessageAgentArgs) string {
	agent, ok := h.Registry.GetAgent(args.InstanceID)
	if !ok {
		return fmt.Sprintf("agent instance '%s' not found", args.InstanceID)
	}
	ctx := context.Background()

	if err := agent.SendUserMessage(ctx, args.Message); err != nil {
		return fmt.Sprintf("failed to send message: %v", err)
	}

	var lastContent string
	recv := agent.Recv()
	for {
		select {
		case res, ok := <-recv:
			if !ok {
				return lastContent
			}
			if res.Chunk != nil && res.Chunk.Done {
				return res.Content
			}
			lastContent = res.Content
		case <-ctx.Done():
			if ctx.Err() != nil {
				return fmt.Sprintf("context cancelled: %v", ctx.Err())
			}
		}
	}
}

func (h *RegistryToolHandler) listAgents(args ListAgentsArgs) string {
	agents := h.Registry.GetRunningAgents()
	if len(agents) == 0 {
		return "No agents currently running."
	}
	var sb strings.Builder
	sb.WriteString("Running agents:\n")
	for id := range agents {
		sb.WriteString(fmt.Sprintf("- %s\n", id))
	}
	return sb.String()
}

func (h *RegistryToolHandler) listAgentTypes(args ListAgentTypesArgs) string {
	h.Registry.mu.RLock()
	defer h.Registry.mu.RUnlock()
	if len(h.Registry.AgentTypes) == 0 {
		return "No agent types registered."
	}
	var sb strings.Builder
	sb.WriteString("Available agent types:\n")
	for name, def := range h.Registry.AgentTypes {
		sb.WriteString(fmt.Sprintf("- %s: %s (%s)\n", name, def.Title, def.Description))
	}
	return sb.String()
}
