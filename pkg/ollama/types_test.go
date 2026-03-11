package ollama

import (
	"encoding/json"
	"testing"
)

func TestChatResponse_Unmarshal(t *testing.T) {
	jsonData := `{
		"model": "gemma3",
		"created_at": "2023-11-07T05:31:56.123456789Z",
		"message": {
			"role": "assistant",
			"content": "the sky is blue",
			"thinking": "analyzing...",
			"tool_calls": [
				{
					"function": {
						"name": "get_weather",
						"arguments": {"location": "London"}
					}
				}
			]
		},
		"done": true,
		"done_reason": "stop",
		"total_duration": 1000000,
		"load_duration": 500000,
		"prompt_eval_count": 10,
		"prompt_eval_duration": 100000,
		"eval_count": 20,
		"eval_duration": 200000,
		"logprobs": [
			{
				"token": "the",
				"logprob": -0.01,
				"bytes": [116, 104, 101],
				"top_logprobs": [
					{"token": "The", "logprob": -1.5, "bytes": [84, 104, 101]}
				]
			}
		]
	}`

	var resp ChatResponse
	if err := json.Unmarshal([]byte(jsonData), &resp); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if resp.Model != "gemma3" {
		t.Errorf("expected model gemma3, got %s", resp.Model)
	}
	if resp.CreatedAt.IsZero() {
		t.Error("expected CreatedAt to be set")
	}
	if resp.Message.Content != "the sky is blue" {
		t.Errorf("expected content, got %s", resp.Message.Content)
	}
	if resp.Message.ReasoningContent != "analyzing..." {
		t.Errorf("expected reasoning content, got %s", resp.Message.ReasoningContent)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	if resp.DoneReason != "stop" {
		t.Errorf("expected done_reason stop, got %s", resp.DoneReason)
	}
	if resp.Logprobs == nil || len(*resp.Logprobs) != 1 {
		t.Errorf("expected 1 logprob, got %d", len(*resp.Logprobs))
	}
	if (*resp.Logprobs)[0].Token != "the" {
		t.Errorf("expected token the, got %s", (*resp.Logprobs)[0].Token)
	}
}

func TestChatRequest_Marshal(t *testing.T) {
	format := ResponseFormat("json")
	keepAlive := "5m"
	logprobs := true
	topLogprobs := 2

	req := ChatRequest{
		BaseRequest: BaseRequest[ChatRequest]{
			Model:       "gemma3",
			Format:      &format,
			KeepAlive:   &keepAlive,
			Think:       true,
			Logprobs:    &logprobs,
			TopLogprobs: &topLogprobs,
		},
		Messages: []Message{
			{Role: "user", Content: "hello"},
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("failed to unmarshal back: %v", err)
	}

	if m["model"] != "gemma3" {
		t.Errorf("expected model gemma3, got %v", m["model"])
	}
	if m["format"] != "json" {
		t.Errorf("expected format json, got %v", m["format"])
	}
	if m["keep_alive"] != "5m" {
		t.Errorf("expected keep_alive 5m, got %v", m["keep_alive"])
	}
	if m["think"] != true {
		t.Errorf("expected think true, got %v", m["think"])
	}
	if m["logprobs"] != true {
		t.Errorf("expected logprobs true, got %v", m["logprobs"])
	}
	if m["top_logprobs"] != float64(2) {
		t.Errorf("expected top_logprobs 2, got %v", m["top_logprobs"])
	}
}

func TestChatRequest_Builder(t *testing.T) {
	req := NewChatRequest("gemma3").
		AddMessage("user", "hello").
		WithStreaming(true).
		WithThinking(true).
		WithKeepAlive("10m")

	if req.Model != "gemma3" {
		t.Errorf("expected model gemma3, got %s", req.Model)
	}
	if len(req.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(req.Messages))
	}
	if req.Messages[0].Content != "hello" {
		t.Errorf("expected content hello, got %s", req.Messages[0].Content)
	}
	if !req.Stream {
		t.Error("expected stream to be true")
	}
	if req.Think != true {
		t.Error("expected think to be true")
	}
	if *req.KeepAlive != "10m" {
		t.Errorf("expected keep_alive 10m, got %s", *req.KeepAlive)
	}
}

func TestChatRequest_BuilderWithOptions(t *testing.T) {
	opts := &ModelOptions{
		Temperature:       0.8,
		ContextWindowSize: 4096,
	}

	req := NewChatRequestWithOptions("gemma3", opts).
		AddMessage("user", "hello")

	if req.Options.Temperature != 0.8 {
		t.Errorf("expected temperature 0.8, got %f", req.Options.Temperature)
	}
	if req.Options.ContextWindowSize != 4096 {
		t.Errorf("expected num_ctx 4096, got %d", req.Options.ContextWindowSize)
	}
}

func TestGenerateRequest_Marshal(t *testing.T) {
	req := NewGenerateRequest("gemma3", "Why is the sky blue?").
		WithSystem("You are a helpful assistant").
		WithStreaming(false).
		WithRaw(true).
		WithContext([]int{1, 2, 3})

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("failed to unmarshal back: %v", err)
	}

	// Verify flat structure (embedded fields should be at top level)
	expected := map[string]interface{}{
		"model":   "gemma3",
		"prompt":  "Why is the sky blue?",
		"system":  "You are a helpful assistant",
		"stream":  false,
		"raw":     true,
		"context": []interface{}{float64(1), float64(2), float64(3)},
	}

	for k := range expected {
		if m[k] == nil {
			t.Errorf("expected key %s to be present", k)
			continue
		}
	}

	if m["model"] != "gemma3" {
		t.Errorf("expected model gemma3, got %v", m["model"])
	}
}

func TestGenerateResponse_Unmarshal(t *testing.T) {
	jsonData := `{
		"model": "gemma3",
		"created_at": "2023-11-07T05:31:56Z",
		"response": "the sky is blue",
		"done": true,
		"context": [1, 2, 3],
		"total_duration": 1000000
	}`

	var resp GenerateResponse
	if err := json.Unmarshal([]byte(jsonData), &resp); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if resp.Model != "gemma3" {
		t.Errorf("expected model gemma3, got %s", resp.Model)
	}
	if resp.Response != "the sky is blue" {
		t.Errorf("expected response, got %s", resp.Response)
	}
	if !resp.Done {
		t.Error("expected done true")
	}
	if len(resp.Context) != 3 || resp.Context[0] != 1 {
		t.Errorf("expected context [1, 2, 3], got %v", resp.Context)
	}
}
