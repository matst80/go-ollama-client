package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

type RegistryEventType string

const (
	EventAgentSpawned RegistryEventType = "agent_spawned"
	EventAgentRemoved RegistryEventType = "agent_removed"
)

type RegistryEvent struct {
	Type          RegistryEventType
	AgentID       string
	ParentAgentID string
}

type AgentDefinition struct {
	spawnFunction func(ctx context.Context, content string) AgentSessionInterface
	Title         string `json:"title"`
	Description   string `json:"description"`
}

func NewAgentDefinition(title, description string, spawnFn func(ctx context.Context, content string) AgentSessionInterface) AgentDefinition {
	return AgentDefinition{
		Title:         title,
		Description:   description,
		spawnFunction: spawnFn,
	}
}

type AgentRegistry struct {
	agents     map[string]AgentSessionInterface
	mu         sync.RWMutex
	AgentTypes map[string]AgentDefinition
	listeners  []func(event RegistryEvent)
}

func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{
		agents:     make(map[string]AgentSessionInterface),
		AgentTypes: make(map[string]AgentDefinition),
		listeners:  make([]func(event RegistryEvent), 0),
	}
}

type AgentConfig struct {
	Name         string   `json:"name"`
	Title        string   `json:"title"`
	Description  string   `json:"description"`
	SystemPrompt string   `json:"system_prompt"`
	Model        string   `json:"model"`
	Client       string   `json:"client"`
	Tools        []string `json:"tools"`
}

type AgentRegistryConfig struct {
	Clients map[string]ChatClientInterface
	Agents  map[string]AgentConfig
	Tools   map[string]ToolDefinition
	OnSpawn func(context.Context, string, AgentSessionInterface)
}

func NewAgentRegistryConfig() *AgentRegistryConfig {
	return &AgentRegistryConfig{
		Clients: make(map[string]ChatClientInterface),
		Agents:  make(map[string]AgentConfig),
		Tools:   make(map[string]ToolDefinition),
	}
}

func (c *AgentRegistryConfig) RegisterClient(name string, client ChatClientInterface) *AgentRegistryConfig {
	c.Clients[name] = client
	return c
}

func (c *AgentRegistryConfig) RegisterTool(tool ToolDefinition) *AgentRegistryConfig {
	c.Tools[tool.Name] = tool
	return c
}

func (c *AgentRegistryConfig) RegisterAgentConfig(config AgentConfig) *AgentRegistryConfig {
	c.Agents[config.Name] = config
	return c
}

func (c *AgentRegistryConfig) LoadAgentsFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var agents map[string]AgentConfig
	if err := json.Unmarshal(data, &agents); err != nil {
		return err
	}
	for _, a := range agents {
		// Ensure name is set if key is name
		if a.Name == "" {
			// If it's a map, we might need to handle it differently, 
			// but the current struct has Name field with json:"name"
		}
		c.Agents[a.Name] = a
	}
	return nil
}

func (c *AgentRegistryConfig) Build() *AgentRegistry {
	registry := NewAgentRegistry()
	for _, agentConfig := range c.Agents {
		config := agentConfig // capture for closure
		registry.RegisterAgent(config.Name, AgentDefinition{
			Title:       config.Title,
			Description: config.Description,
			spawnFunction: func(ctx context.Context, content string) AgentSessionInterface {
				client := c.Clients[config.Client]
				// Note: If client is nil, session will be created with nil client.
				// ChatStreamed will fail later, which is acceptable if misconfigured.

				req := NewChatRequest(config.Model)
				if config.SystemPrompt != "" {
					req.Messages = []Message{{Role: MessageRoleSystem, Content: config.SystemPrompt}}
				}
				if content != "" {
					req.Messages = append(req.Messages, Message{Role: MessageRoleUser, Content: content})
				}

				// Add tools
				for _, toolName := range config.Tools {
					if toolDef, ok := c.Tools[toolName]; ok {
						req.AddTool(toolDef.ToTool())
					}
				}

				session := NewAgentSession(ctx, client, req)

				// Apply OnSpawn hook from config
				if c.OnSpawn != nil {
					c.OnSpawn(ctx, config.Name, session)
				}

				return session
			},
		})
	}
	return registry
}

func (r *AgentRegistry) AddEventListener(listener func(event RegistryEvent)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.listeners = append(r.listeners, listener)
}

func (r *AgentRegistry) emitEvent(event RegistryEvent) {
	var listeners []func(event RegistryEvent)
	r.mu.RLock()
	listeners = append(listeners, r.listeners...)
	r.mu.RUnlock()

	for _, listener := range listeners {
		listener(event)
	}
}

func (r *AgentRegistry) RegisterAgent(name string, agent AgentDefinition) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.AgentTypes[name] = agent
}

func (r *AgentRegistry) GetAgentType(name string) (AgentDefinition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agent, ok := r.AgentTypes[name]
	return agent, ok
}

func (r *AgentRegistry) GetRunningAgents() map[string]AgentSessionInterface {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agents := make(map[string]AgentSessionInterface)
	for k, v := range r.agents {
		agents[k] = v
	}
	return agents
}

func (r *AgentRegistry) GetAgent(instanceID string) (AgentSessionInterface, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agent, ok := r.agents[instanceID]
	return agent, ok
}

func (r *AgentRegistry) SpawnAgent(ctx context.Context, typeName string, instanceID string, content string, parentID string) (AgentSessionInterface, error) {
	r.mu.Lock()
	agentDef, ok := r.AgentTypes[typeName]
	if !ok {
		r.mu.Unlock()
		return nil, fmt.Errorf("agent type %s not found", typeName)
	}

	if _, exists := r.agents[instanceID]; exists {
		r.mu.Unlock()
		return nil, fmt.Errorf("agent instance %s already exists", instanceID)
	}

	session := agentDef.spawnFunction(ctx, content)
	session.SetState(func(s *AgentState) {
		s.Title = agentDef.Title
		s.Type = typeName
		s.ParentID = parentID
	})
	r.agents[instanceID] = session
	r.mu.Unlock()

	r.emitEvent(RegistryEvent{
		Type:          EventAgentSpawned,
		AgentID:       instanceID,
		ParentAgentID: parentID,
	})

	return session, nil
}

func (r *AgentRegistry) RemoveAgent(instanceID string) {
	r.mu.Lock()
	session, ok := r.agents[instanceID]
	if ok {
		delete(r.agents, instanceID)
	}
	r.mu.Unlock()

	if ok {
		session.Stop()
		r.emitEvent(RegistryEvent{
			Type:    EventAgentRemoved,
			AgentID: instanceID,
		})
	}
}
