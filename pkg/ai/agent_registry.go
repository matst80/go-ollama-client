package ai

import (
	"context"
	"fmt"
	"sync"
)

type RegistryEventType string

const (
	EventAgentSpawned RegistryEventType = "agent_spawned"
	EventAgentRemoved RegistryEventType = "agent_removed"
)

type RegistryEvent struct {
	Type    RegistryEventType
	AgentID string
}

type AgentDefinition struct {
	Title         string
	Description   string
	SpawnFunction func(ctx context.Context, content string) AgentSessionInterface
}

func NewAgentDefinition(title, description string, spawnFn func(ctx context.Context, content string) AgentSessionInterface) AgentDefinition {
	return AgentDefinition{
		Title:         title,
		Description:   description,
		SpawnFunction: spawnFn,
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

func (r *AgentRegistry) SpawnAgent(ctx context.Context, typeName string, instanceID string, content string) (AgentSessionInterface, error) {
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

	session := agentDef.SpawnFunction(ctx, content)
	r.agents[instanceID] = session
	r.mu.Unlock()

	r.emitEvent(RegistryEvent{
		Type:    EventAgentSpawned,
		AgentID: instanceID,
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
