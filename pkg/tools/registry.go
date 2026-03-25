package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/matst80/go-ai-agent/pkg/ai"
)

// Registry maintains a list of tools that can be used by the model
type Registry struct {
	tools map[string]*ai.ToolDefinition
	Tools []ai.Function
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]*ai.ToolDefinition),
	}
}

func (r *Registry) HasTool(name string) bool {
	_, ok := r.tools[name]
	return ok
}

// Register adds a new tool to the registry.
// name: The name of the tool.
// args: A pointer to a struct that defines the tool's parameters.
// fn: The function to call when the tool is invoked.
func (r *Registry) Register(name, description string, args any, fn any) error {
	toolDef, err := ai.GetToolDefinition(name, description, args, fn)
	if err != nil {
		return err
	}
	// turbo-all
	r.tools[name] = toolDef

	return nil
}

func (r *Registry) RegisterTool(tool ai.ToolDefinition) error {
	if _, ok := r.tools[tool.Name]; ok {
		return fmt.Errorf("tool %s already registered", tool.Name)
	}
	r.tools[tool.Name] = &tool
	return nil
}

func (r *Registry) RegisterTools(defs ...ai.ToolDefinition) {
	for _, tool := range defs {
		r.RegisterTool(tool)
	}
}

func (r *Registry) GetTool(name string) (*ai.ToolDefinition, bool) {
	def, ok := r.tools[name]
	return def, ok
}

// GetTools returns the tools in a format generic format
func (r *Registry) GetTools() []ai.Tool {
	var tools []ai.Tool
	for _, def := range r.tools {
		if !def.Enabled {
			continue
		}
		tools = append(tools, ai.Tool{
			Type: ai.ToolTypeFunction,
			Function: ai.Function{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.Parameters,
			},
		})
	}
	return tools
}

// Call invokes a registered tool by name with the given JSON arguments.
// The arguments should be a JSON object mapping field names to values.
func (r *Registry) Call(ctx context.Context, name string, argsJSON json.RawMessage) ([]reflect.Value, error) {
	def, ok := r.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool %s not found", name)
	}

	// Create a new instance of the args struct
	argsPtr := reflect.New(def.ArgsType)

	// Unmarshal the JSON into the struct
	if err := json.Unmarshal(argsJSON, argsPtr.Interface()); err != nil {
		return nil, fmt.Errorf("failed to unmarshal arguments for tool %s: %v", name, err)
	}

	// Prepare arguments for the call
	var callArgs []reflect.Value
	handlerType := def.Handler.Type()

	// Check if the handler expects a context.Context as the first argument
	if handlerType.NumIn() > 0 && handlerType.In(0).String() == "context.Context" {
		callArgs = append(callArgs, reflect.ValueOf(ctx))
	}

	// Add the args struct (Elem() of the pointer)
	// Note: We assume the last (or second) argument is the args struct
	callArgs = append(callArgs, argsPtr.Elem())

	// Call the handler
	results := def.Handler.Call(callArgs)

	return results, nil
}
