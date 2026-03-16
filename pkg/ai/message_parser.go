package ai

import (
	"context"
)

// MessageParser parses AccumulatedResponse chunks into StreamMessage objects.
// Implementations may be stateful (e.g., fence parser) or stateless (NDJSON parser).
type MessageParser interface {
	Parse(ctx context.Context, res *AccumulatedResponse) ([]*StreamMessage, error)
}
