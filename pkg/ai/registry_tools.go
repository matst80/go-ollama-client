package ai

import (
	"context"
	"fmt"
	"strings"
)

// RegistryToolHandler provides tools for agents to interact with the AgentRegistry
type RegistryToolHandler struct {
	Registry    *AgentRegistry
	createState func(ctx context.Context, content string) AgentState
	tools       map[string]ToolDefinition
}

func NewRegistryToolHandler(registry *AgentRegistry, createState func(ctx context.Context, content string) AgentState) *RegistryToolHandler {
	h := &RegistryToolHandler{
		Registry:    registry,
		createState: createState,
		tools:       make(map[string]ToolDefinition),
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

// AgentStatusArgs requests status information for a running agent instance.
type AgentStatusArgs struct {
	InstanceID string `json:"instance_id" tool:"The ID of the running agent instance,required"`
}

type ListAgentsArgs struct{}

type ListAgentTypesArgs struct{}

type ReportArgs struct {
	Message string `json:"message" tool:"The message to report back to the parent agent.,required"`
}

func (h *RegistryToolHandler) initTools() {
	tools := []struct {
		name string
		desc string
		args any
		fn   any
	}{
		{"spawn_agent", "Spawn a new agent instance of a given type. Use unique instance_id.", SpawnAgentArgs{}, h.spawnAgent},
		{"message_agent", "Send a message to a running agent instance and wait for response.", MessageAgentArgs{}, h.messageAgent},
		{"report", "Report a message back to the parent agent.", ReportArgs{}, h.report},
		{"agent_status", "Get status of a running agent instance.", AgentStatusArgs{}, h.agentStatus},
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

func (h *RegistryToolHandler) spawnAgent(ctx context.Context, args SpawnAgentArgs) string {
	state := h.createState(ctx, args.Content)
	_, err := h.Registry.SpawnAgent(ctx, args.TypeName, args.InstanceID, args.Content, state)
	if err != nil {
		return fmt.Sprintf("failed to spawn agent: %v", err)
	}
	return fmt.Sprintf("Agent instance '%s' spawned successfully.", args.InstanceID)
}

func (h *RegistryToolHandler) report(ctx context.Context, args ReportArgs) string {
	agentID, ok := ctx.Value("agentID").(string)
	if !ok {
		return "error: could not determine current agent ID"
	}

	agent, ok := h.Registry.GetAgent(agentID)
	if !ok {
		return "error: current agent not found in registry"
	}

	parentID := agent.GetState().GetParentID()
	if parentID == "" {
		return "error: no parent agent to report to"
	}

	parent, ok := h.Registry.GetAgent(parentID)
	if !ok {
		return fmt.Sprintf("error: parent agent '%s' not found", parentID)
	}

	reportMsg := fmt.Sprintf("Report from %s: %s", agentID, args.Message)
	if err := parent.SendUserMessage(ctx, reportMsg); err != nil {
		return fmt.Sprintf("failed to send report to parent: %v", err)
	}

	return "Report sent to parent successfully."
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

// agentStatus returns a human-readable status summary for a running agent instance.
func (h *RegistryToolHandler) agentStatus(args AgentStatusArgs) string {
	agent, ok := h.Registry.GetAgent(args.InstanceID)
	if !ok {
		return fmt.Sprintf("agent instance '%s' not found", args.InstanceID)
	}

	state := agent.GetState()
	msgs := agent.GetMessageHistory()

	var sb strings.Builder
	fmt.Fprintf(&sb, "Agent %s status:\n", args.InstanceID)
	fmt.Fprintf(&sb, "  Title: %s\n", state.GetTitle())
	fmt.Fprintf(&sb, "  Type: %s\n", state.GetType())
	fmt.Fprintf(&sb, "  Status: %s\n", state.GetStatus())
	fmt.Fprintf(&sb, "  ParentID: %s\n", state.GetParentID())
	fmt.Fprintf(&sb, "  CreatedAt: %v\n", state.GetCreatedAt())
	fmt.Fprintf(&sb, "  LastActive: %v\n", state.GetLastActive())
	fmt.Fprintf(&sb, "  Model: %s\n", agent.GetModel())
	fmt.Fprintf(&sb, "  MessageCount: %d\n", len(msgs))
	if len(msgs) > 0 {
		last := msgs[len(msgs)-1]
		preview := strings.TrimSpace(last.Content)
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		fmt.Fprintf(&sb, "  LastMessageRole: %s\n", last.Role)
		fmt.Fprintf(&sb, "  LastMessagePreview: %s\n", preview)
	}

	return sb.String()
}

func (h *RegistryToolHandler) listAgents(args ListAgentsArgs) string {
	agents := h.Registry.GetRunningAgents()
	if len(agents) == 0 {
		return "No agents currently running."
	}
	var sb strings.Builder
	sb.WriteString("Running agents:\n")
	for id := range agents {
		fmt.Fprintf(&sb, "- %s\n", id)
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
