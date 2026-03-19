# MCP Proxy Testing

The application now supports loading MCP servers from a configuration file.

## Configuration

Create a file named `mcp_config.json` in the root directory:

```json
{
  "mcpServers": {
    "browsermcp": {
      "command": "npx",
      "args": ["-y", "@browsermcp/mcp@latest"],
      "env": {}
    }
  }
}
```

## How to Test

1. Ensure you have `node` and `npx` installed.
2. Run the application: `go run main.go`.
3. The application will log the registered tools from the configured MCP servers.
4. The master agent will have access to these tools.

## Example Tools from Browser MCP

- `browser_navigate`: Navigate to a URL.
- `browser_screenshot`: Take a screenshot of the current page.
- `browser_type`: Type text into an element.
- `browser_click`: Click on an element.
