package ai

import (
	"context"
	"fmt"
	"sync"
)

type AgentSessionInterface interface {
	SendUserMessage(ctx context.Context, msg string) error
	RecvAny() <-chan any
	Stop()
}

type AgentDefinition struct {
	Title         string
	Description   string
	SpawnFunction func(ctx context.Context, content string) AgentSessionInterface
}

type AgentRegistry struct {
	agents     map[string]AgentSessionInterface
	mu         sync.RWMutex
	AgentTypes map[string]AgentDefinition
}

func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{
		agents:     make(map[string]AgentSessionInterface),
		AgentTypes: make(map[string]AgentDefinition),
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

func (r *AgentRegistry) SpawnAgent(ctx context.Context, typeName string, instanceID string, content string) (AgentSessionInterface, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	agentDef, ok := r.AgentTypes[typeName]
	if !ok {
		return nil, fmt.Errorf("agent type %s not found", typeName)
	}

	if _, exists := r.agents[instanceID]; exists {
		return nil, fmt.Errorf("agent instance %s already exists", instanceID)
	}

	session := agentDef.SpawnFunction(ctx, content)
	r.agents[instanceID] = session
	return session, nil
}


func (r *AgentRegistry) RemoveAgent(instanceID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if session, ok := r.agents[instanceID]; ok {
		session.Stop()
		delete(r.agents, instanceID)
	}
}

