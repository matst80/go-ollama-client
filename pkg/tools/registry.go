package tools

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/matst80/go-ollama-client/pkg/ai"
)

// Registry maintains a list of tools that can be used by the model
type Registry struct {
	tools map[string]ToolDefinition
	Tools []ai.Function
}

// ToolDefinition represents a registered tool
type ToolDefinition struct {
	parameters map[string]interface{}
	Name       string
	Enabled    bool
	ArgsType   reflect.Type
	Handler    reflect.Value
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]ToolDefinition),
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
func (r *Registry) Register(name string, args any, fn any) error {
	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func {
		return fmt.Errorf("handler must be a function")
	}

	argType := reflect.TypeOf(args)
	if argType.Kind() == reflect.Ptr {
		argType = argType.Elem()
	}

	if argType.Kind() != reflect.Struct {
		return fmt.Errorf("args must be a struct or a pointer to a struct")
	}

	r.tools[name] = ToolDefinition{
		Name:       name,
		Enabled:    true,
		ArgsType:   argType,
		Handler:    v,
		parameters: generateJSONSchema(argType),
	}

	return nil
}

// GetTools returns the tools in a format compatible with Ollama's API
func (r *Registry) GetTools() []ai.Tool {
	var tools []ai.Tool
	for _, def := range r.tools {
		// Since we don't have a tool-level description in Register,
		// we use the name as a placeholder or could potentially
		// extract it from a field tag if we wanted to be creative.
		// For now, let's just use the name.
		if !def.Enabled {
			continue
		}
		tools = append(tools, ai.Tool{
			Type: ai.ToolTypeFunction,
			Function: ai.Function{
				Name:        def.Name,
				Description: def.Name, // Default to name
				Parameters:  def.parameters,
			},
		})
	}
	return tools
}

// Call invokes a registered tool by name with the given JSON arguments.
// The arguments should be a JSON object mapping field names to values.
func (r *Registry) Call(name string, argsJSON json.RawMessage) ([]reflect.Value, error) {
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

	// Call the handler with the unmarshaled struct
	// Note: We assume the handler takes the args struct (not pointer) as its ONLY argument
	// based on the user's example: func RunCommand(args RunArgs)
	results := def.Handler.Call([]reflect.Value{argsPtr.Elem()})

	return results, nil
}
