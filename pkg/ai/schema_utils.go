package ai

import (
	"reflect"
	"strings"
)

func generateJSONSchema(t reflect.Type) map[string]interface{} {
	schema := map[string]interface{}{
		"type":       "object",
		"properties": make(map[string]interface{}),
	}

	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" || jsonTag == "" {
			continue
		}

		// Handle json:"name,omitempty"
		name := strings.Split(jsonTag, ",")[0]

		toolTag := field.Tag.Get("tool")
		description := ""
		isRequired := false

		if toolTag != "" {
			parts := strings.Split(toolTag, ",")
			description = parts[0]
			for _, p := range parts[1:] {
				if strings.TrimSpace(p) == "required" {
					isRequired = true
				}
			}
		}

		if isRequired {
			required = append(required, name)
		}

		property := map[string]interface{}{
			"type": goTypeToJSONType(field.Type),
		}
		if description != "" {
			property["description"] = description
		}

		schema["properties"].(map[string]interface{})[name] = property
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

func goTypeToJSONType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Bool:
		return "boolean"
	case reflect.Slice, reflect.Array:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	case reflect.Ptr:
		return goTypeToJSONType(t.Elem())
	default:
		return "string"
	}
}
