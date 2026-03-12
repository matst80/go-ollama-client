package ai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
)

// TestChunkReader_NormalLines verifies that ChunkReader reads multiple newline-delimited lines
// and invokes the handler for each trimmed non-empty line.
func TestChunkReader_NormalLines(t *testing.T) {
	data := "first line\nsecond line\nthird line\n"
	r := bytes.NewBufferString(data)

	var got [][]byte
	handler := func(line []byte) (stop bool) {
		// copy since the slice may be reused by the buffer
		c := make([]byte, len(line))
		copy(c, line)
		got = append(got, c)
		return false
	}

	if err := ChunkReader(context.Background(), r, handler); err != nil {
		t.Fatalf("ChunkReader returned error: %v", err)
	}

	if len(got) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(got))
	}
	if string(got[0]) != "first line" || string(got[1]) != "second line" || string(got[2]) != "third line" {
		t.Fatalf("unexpected lines: %q", got)
	}
}

// TestChunkReader_EmptyLinesIgnored ensures that whitespace-only lines are ignored.
func TestChunkReader_EmptyLinesIgnored(t *testing.T) {
	data := "   \n\n  \nonly\n \n"
	r := bytes.NewBufferString(data)

	var got [][]byte
	handler := func(line []byte) (stop bool) {
		c := make([]byte, len(line))
		copy(c, line)
		got = append(got, c)
		return false
	}

	if err := ChunkReader(context.Background(), r, handler); err != nil {
		t.Fatalf("ChunkReader returned error: %v", err)
	}

	if len(got) != 1 {
		t.Fatalf("expected 1 non-empty line, got %d", len(got))
	}
	if string(got[0]) != "only" {
		t.Fatalf("unexpected content: %q", got[0])
	}
}

// TestChunkReader_PartialEOF verifies that data without a trailing newline is processed.
func TestChunkReader_PartialEOF(t *testing.T) {
	data := "partial-without-newline"
	r := bytes.NewBufferString(data)

	var got [][]byte
	handler := func(line []byte) (stop bool) {
		c := make([]byte, len(line))
		copy(c, line)
		got = append(got, c)
		return false
	}

	if err := ChunkReader(context.Background(), r, handler); err != nil {
		t.Fatalf("ChunkReader returned error: %v", err)
	}

	if len(got) != 1 {
		t.Fatalf("expected 1 line, got %d", len(got))
	}
	if string(got[0]) != data {
		t.Fatalf("unexpected content: %q", got[0])
	}
}

// TestChunkReader_StopEarly verifies that returning true from the handler stops the reader early.
func TestChunkReader_StopEarly(t *testing.T) {
	data := "a\nb\nc\n"
	r := bytes.NewBufferString(data)

	var got [][]byte
	handler := func(line []byte) (stop bool) {
		c := make([]byte, len(line))
		copy(c, line)
		got = append(got, c)
		// stop when we see "b"
		if string(line) == "b" {
			return true
		}
		return false
	}

	if err := ChunkReader(context.Background(), r, handler); err != nil {
		t.Fatalf("ChunkReader returned error: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("expected to stop after 2 lines, got %d", len(got))
	}
	if string(got[0]) != "a" || string(got[1]) != "b" {
		t.Fatalf("unexpected lines: %q", got)
	}
}

// TestChunkReader_CancelledContext verifies that ChunkReader returns context.Canceled
// if the context is cancelled by the handler during processing.
func TestChunkReader_CancelledContext(t *testing.T) {
	pr, pw := io.Pipe()

	// write some lines and then close writer in a goroutine
	go func() {
		_, _ = pw.Write([]byte("one\n"))
		_, _ = pw.Write([]byte("two\n"))
		_ = pw.Close()
	}()

	ctx, cancel := context.WithCancel(context.Background())

	var got [][]byte
	handler := func(line []byte) (stop bool) {
		c := make([]byte, len(line))
		copy(c, line)
		got = append(got, c)
		// cancel the context once the first line is received
		if string(line) == "one" {
			cancel()
		}
		return false
	}

	err := ChunkReader(ctx, pr, handler)
	if err == nil {
		t.Fatalf("expected an error due to cancelled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", err)
	}

	// ensure at least the first line was observed, and second may or may not be depending on timing
	if len(got) == 0 || string(got[0]) != "one" {
		t.Fatalf("expected first line 'one' to be received before cancellation, got: %q", got)
	}
}
