package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPullModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/pull" {
			t.Errorf("expected /api/pull, got %s", r.URL.Path)
		}
		var req PushPullRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.Model != "test-model" {
			t.Errorf("expected test-model, got %s", req.Model)
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(StatusResponse{Status: "success"})
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	resp, err := client.PullModel(context.Background(), *NewPushPullRequest("test-model"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != "success" {
		t.Errorf("expected success, got %s", resp.Status)
	}
}

func TestPullModelStreamed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)

		responses := []StatusResponse{
			{Status: "pulling manifest"},
			{Status: "downloading digest"},
			{Status: "success"},
		}

		for _, res := range responses {
			json.NewEncoder(w).Encode(res)
			w.(http.Flusher).Flush()
		}
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	ch := make(chan *StatusResponse)

	go func() {
		err := client.PullModelStreamed(context.Background(), *NewPushPullRequest("test-model"), ch)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}()

	var received []string
	for res := range ch {
		received = append(received, res.Status)
	}

	expected := []string{"pulling manifest", "downloading digest", "success"}
	if len(received) != len(expected) {
		t.Fatalf("expected %d messages, got %d", len(expected), len(received))
	}
	for i, v := range expected {
		if received[i] != v {
			t.Errorf("expected %s at index %d, got %s", v, i, received[i])
		}
	}
}

func TestDeleteModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/delete" {
			t.Errorf("expected /api/delete, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	err := client.DeleteModel(context.Background(), "test-model")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGetVersion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/version" {
			t.Errorf("expected /api/version, got %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(VersionResponse{Version: "0.1.23"})
	}))
	defer server.Close()

	client := NewOllamaClient(server.URL)
	version, err := client.GetVersion(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if version != "0.1.23" {
		t.Errorf("expected 0.1.23, got %s", version)
	}
}
