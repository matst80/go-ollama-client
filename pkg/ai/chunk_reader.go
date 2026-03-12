package ai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
)

// ChunkHandler is a callback invoked for each non-empty, trimmed line read
// from a streaming HTTP response (or any io.Reader delivering newline
// delimited chunks). The handler receives the cleaned line bytes. If it
// returns true the reader will stop processing further lines.
type ChunkHandler func(line []byte) (stop bool)

func JsonChunkReader[T any](data *T, handler func(*T) bool) ChunkHandler {
	return func(line []byte) (stop bool) {
		//log.Println(string(line))
		if err := json.Unmarshal(line, data); err != nil {
			// skip malformed chunk
			log.Printf("error parsing %s", line)
			return false
		}
		return handler(data)
	}
}

var DATA_PREFIX = []byte("data: ")
var DONE = []byte("[DONE]")
var DATA_DONE = []byte("data: [DONE]")
var DATA_PREFIX_LEN = len(DATA_PREFIX)

func DataJsonChunkReader[T any](data *T, handler func(*T) bool) ChunkHandler {
	return func(input []byte) (stop bool) {
		if bytes.Equal(input, DONE) || bytes.Equal(input, DATA_DONE) {
			return true
		}

		// Expect lines to start with the data prefix
		if !bytes.HasPrefix(input, DATA_PREFIX) {
			return false
		}

		if err := json.Unmarshal(input[DATA_PREFIX_LEN:], data); err != nil {
			log.Printf("error parsing: %s, err: %s", input, err)
			return false
		}
		return handler(data)
	}
}

// ChunkReader reads newline-delimited chunks from the provided reader and
// invokes the given handler for each non-empty trimmed line. It respects the
// provided context for cancellation and returns an error if reading fails or
// if the context is canceled.
//
// Behavior notes:
//   - Lines are read using a bufio.Reader and split on '\n'.
//   - Each line is trimmed with bytes.TrimSpace before being passed to the handler.
//   - Empty/whitespace-only lines are ignored.
//   - If the underlying ReadBytes returns data with io.EOF, that final data is
//     still processed before returning nil.
func ChunkReader(ctx context.Context, r io.Reader, handler ChunkHandler) error {
	reader := bufio.NewReader(r)

	for {
		// Check for context cancellation before attempting to read.
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		// If we got data even with an error (like io.EOF), we'll still want to process it.
		clean := bytes.TrimSpace(line)
		if len(clean) > 0 {
			// If handler returns true, stop reading further chunks.
			if handler(clean) {
				return nil
			}
		}

		if err != nil {
			// If the context was canceled concurrently, prefer that error.
			if ctx.Err() != nil {
				return ctx.Err()
			}
			// If EOF, we've already processed any remaining data above and can exit cleanly.
			if err == io.EOF {
				return nil
			}
			// Otherwise propagate the read error.
			return err
		}
	}
}
