package ai

import (
	"context"
)

// StreamedBlock represents a single fenced streamed block extracted from
// assistant output. The Content field must preserve the raw block body exactly.
type StreamedBlock struct {
	Type    string
	Content string
	Done    bool
}

// BlockParser parses accumulated streamed assistant output into typed blocks.
// Implementations may be stateful to support content arriving across multiple
// streamed chunks.
type BlockParser interface {
	ParseBlocks(ctx context.Context, res *AccumulatedResponse) ([]*StreamedBlock, error)
}

// BlockHandler processes parsed streamed blocks.
// Implementations can choose to ignore block types they do not handle.
type BlockHandler interface {
	HandleBlock(ctx context.Context, block *StreamedBlock) error
}
