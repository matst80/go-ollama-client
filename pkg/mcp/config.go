package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/matst80/go-ai-agent/pkg/tools"
)

// ServerConfig represents the configuration for a single MCP server.
type ServerConfig struct {
	Command       string            `json:"command"`
	Args          []string          `json:"args"`
	Env           map[string]string `json:"env"`
	DisabledTools []string          `json:"disabledTools"`
}

// ConfigFile represents the standard MCP configuration file format.
type ConfigFile struct {
	Servers map[string]ServerConfig `json:"-"`
}

// UnmarshalJSON custom unmarshaler for ConfigFile since the root is an object mapping names to configs
// Note: Depending on the format, it might be nested under "mcpServers" like in Claude Config.
// Let's assume the provided JSON is just a map of strings to ServerConfig.
func (c *ConfigFile) UnmarshalJSON(data []byte) error {
	// Try Claude's format first (nested under "mcpServers")
	var claudeFormat struct {
		MCPServers map[string]ServerConfig `json:"mcpServers"`
	}
	if err := json.Unmarshal(data, &claudeFormat); err == nil && len(claudeFormat.MCPServers) > 0 {
		c.Servers = claudeFormat.MCPServers
		return nil
	}

	// Try raw map format
	var raw map[string]ServerConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	c.Servers = raw
	return nil
}

// LoadConfig reads an MCP configuration file and loads the servers.
func LoadConfig(path string) (map[string]ServerConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var conf ConfigFile
	if err := json.Unmarshal(data, &conf); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return conf.Servers, nil
}

// LoadAndRegisterServers reads the config and registers all servers onto the given registry.
func LoadAndRegisterServers(ctx context.Context, configPath string, registry *tools.Registry) ([]*ServerProxy, error) {
	servers, err := LoadConfig(configPath)
	if err != nil {
		return nil, err
	}

	var proxies []*ServerProxy

	for name, cfg := range servers {
		// Convert env map to slice of "K=V"
		var envVars []string
		for k, v := range cfg.Env {
			envVars = append(envVars, fmt.Sprintf("%s=%s", k, v))
		}
		// Also append current system env if needed, or pass just what's defined.
		// For safety, let's include system environments as well, then overwrite with config envs.
		// A cleaner approach is to use the environment slice directly in the client.
		sysEnv := os.Environ()
		sysEnv = append(sysEnv, envVars...)

		proxy, err := NewStdioServerProxy(cfg.Command, sysEnv, cfg.Args...)
		if err != nil {
			return proxies, fmt.Errorf("failed to start server %s: %w", name, err)
		}

		if err := proxy.Initialize(ctx); err != nil {
			proxy.Close()
			return proxies, fmt.Errorf("failed to initialize server %s: %w", name, err)
		}

		// Prepare disabled tools map for fast lookup
		disabled := make(map[string]bool)
		for _, dt := range cfg.DisabledTools {
			disabled[dt] = true
		}

		// Register tools using the custom method that filters
		if err := proxy.RegisterToolsWithFilter(ctx, registry, disabled); err != nil {
			proxy.Close()
			return proxies, fmt.Errorf("failed to register tools for %s: %w", name, err)
		}

		proxies = append(proxies, proxy)
	}

	return proxies, nil
}
