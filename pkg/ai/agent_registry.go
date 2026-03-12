package ai

import (
	"context"
	"fmt"
	"sync"
)

type AgentDefinition struct {
	Title         string
	Description   string
	SpawnFunction func(ctx context.Context, content string) *AgentSession[any]
}

type AgentRegistry struct {
	agents     map[string]*AgentSession[any]
	mu         sync.RWMutex
	AgentTypes map[string]AgentDefinition
}

func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{
		agents: make(map[string]*AgentSession[any]),
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

func (r *AgentRegistry) GetRunningAgents() []*AgentSession[any] {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agents := make([]*AgentSession[any], 0, len(r.agents))
	for _, agent := range r.agents {
		agents = append(agents, agent)
	}
	return agents
}

func (r *AgentRegistry) SpawnAgent(ctx context.Context, name string, content string) (*AgentSession[any], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agent, ok := r.AgentTypes[name]
	if !ok {
		return nil, fmt.Errorf("agent %s not found", name)
	}
	return agent.SpawnFunction(ctx, content), nil
}
