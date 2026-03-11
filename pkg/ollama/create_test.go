package ollama

import (
	"testing"
)

func TestCreateRequest(t *testing.T) {
	req := NewCreateRequest("test-model").
		WithFrom("llama2").
		WithModelfile("FROM llama2\nSYSTEM You are a helpful assistant.").
		WithStreaming(false)

	if req.Model != "test-model" {
		t.Errorf("expected model test-model, got %s", req.Model)
	}
	if req.From != "llama2" {
		t.Errorf("expected from llama2, got %s", req.From)
	}
	if req.Stream != false {
		t.Error("expected stream to be false")
	}
}
