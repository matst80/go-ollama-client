package ai

import (
	"fmt"
	"testing"
)

// This test demonstrates why returning an error from StreamWorker
// might lead to lost errors if not handled carefully by the caller.
func TestStreamWorker_PotentialIssue(t *testing.T) {
	// Let's assume the proposed signature:
	// StreamWorker(ctx context.Context, req ChatRequest, ch chan *ChatResponse) error

	// Mocking what would happen in a real scenario:
	streamWorkerMock := func(ch chan *ChatResponse) error {
		defer close(ch)
		ch <- &ChatResponse{Message: Message{Content: "Hello"}}
		ch <- &ChatResponse{Message: Message{Content: " "}}
		// An error occurs mid-stream
		return fmt.Errorf("network timeout mid-stream")
	}

	ch := make(chan *ChatResponse)

	// Issue 1: Using 'go' ignores the return error
	go streamWorkerMock(ch)

	for resp := range ch {
		fmt.Printf("Received: %s\n", resp.Message.Content)
	}

	// At this point, the loop finished because the channel closed.
	// But did it finish successfully? We don't know because the
	// return error from streamWorkerMock was ignored by the 'go' statement.
}

// Correct way to handle it if returning an error:
func TestStreamWorker_CorrectHandling(t *testing.T) {
	streamWorkerMock := func(ch chan *ChatResponse) error {
		defer close(ch)
		ch <- &ChatResponse{Message: Message{Content: "Hello"}}
		return fmt.Errorf("failed")
	}

	ch := make(chan *ChatResponse)
	errCh := make(chan error, 1)

	go func() {
		errCh <- streamWorkerMock(ch)
	}()

	for resp := range ch {
		_ = resp
	}

	if err := <-errCh; err != nil {
		fmt.Printf("Detected error: %v\n", err)
	}
}
