package ai

import (
	"context"
	"fmt"
	"strings"
)

// FenceParser parses streamed fenced blocks using the exact form:
//
// ```type
// ...block body...
// ```
type FenceParser struct {
	buf            strings.Builder
	EmitFragments  bool
}

func NewFenceParser() *FenceParser {
	return &FenceParser{
		EmitFragments: true,
	}
}

// ParseBlocks accepts accumulated streamed content and emits StreamedBlock values
// whenever complete fenced blocks have been received.
func (p *FenceParser) ParseBlocks(ctx context.Context, res *AccumulatedResponse) ([]*StreamedBlock, error) {
	_ = ctx

	if res == nil || res.Chunk == nil {
		return nil, nil
	}

	p.buf.WriteString(res.Chunk.Message.Content)
	input := p.buf.String()

	var out []*StreamedBlock

	for {
		start := strings.Index(input, "```")
		if start == -1 {
			break
		}

		newlineOffset := strings.IndexByte(input[start:], '\n')
		if newlineOffset == -1 {
			break
		}
		newlineOffset += start

		header := strings.TrimSpace(input[start+3 : newlineOffset])
		if header == "" || strings.ContainsAny(header, " \t") || strings.ContainsAny(header, "`") {
			input = input[start+3:]
			continue
		}

		bodyAndTail := input[newlineOffset+1:]
		endOffset := strings.Index(bodyAndTail, "```")
		if endOffset == -1 {
			if res.Chunk.Done || p.EmitFragments {
				// Emit fragment (intermediate or final)
				out = append(out, &StreamedBlock{
					Type:    header,
					Content: bodyAndTail,
					Done:    res.Chunk.Done,
				})
				if res.Chunk.Done {
					input = ""
				}
			}
			break
		}

		body := bodyAndTail[:endOffset]
		// Handle optional newline directly before backticks
		if len(body) > 0 && body[len(body)-1] == '\n' {
			body = body[:len(body)-1]
		}

		out = append(out, &StreamedBlock{
			Type:    header,
			Content: body,
			Done:    true,
		})

		input = bodyAndTail[endOffset+len("```"):]
	}

	p.buf.Reset()
	p.buf.WriteString(input)

	return out, nil
}

// AttachBlockParserToAccumulator feeds accumulated assistant output into the
// block parser, dispatches any parsed blocks to the provided handler, and
// forwards the original accumulated responses unchanged.
func AttachBlockParserToAccumulator(
	ctx context.Context,
	input <-chan *AccumulatedResponse,
	parser BlockParser,
	handler BlockHandler,
) <-chan *AccumulatedResponse {
	out := make(chan *AccumulatedResponse)

	go func() {
		defer close(out)

		for res := range input {
			if res != nil && parser != nil && handler != nil {
				blocks, err := parser.ParseBlocks(ctx, res)
				if err != nil {
					fmt.Printf("block parser error: %v\n", err)
				} else {
					for _, block := range blocks {
						if err := handler.HandleBlock(ctx, block); err != nil {
							fmt.Printf("block handler error: %v\n", err)
						}
					}
				}
			}

			select {
			case out <- res:
			case <-ctx.Done():
				return
			}
		}
	}()

	return out
}
