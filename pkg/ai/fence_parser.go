package ai

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// FenceParser parses fenced diffstream markdown blocks spanning arbitrary chunks.
type FenceParser struct {
	buf strings.Builder
}

func NewFenceParser() *FenceParser { return &FenceParser{} }

func atoiOrZero(s string) int {
	i, _ := strconv.Atoi(s)
	return i
}

func trimQuotes(s string) string {
	return strings.Trim(s, `"'`)
}

func parseHeaderAttrs(raw string) map[string]string {
	m := map[string]string{}
	// support quoted values and values containing spaces by scanning
	// tokens but allowing "key=\"a b c\"" style quoted values.
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return m
	}
	parts := []string{}
	cur := ""
	inQuotes := false
	quoteChar := byte(0)
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inQuotes {
			cur += string(c)
			if c == quoteChar {
				inQuotes = false
			}
			continue
		}
		if c == '\'' || c == '"' {
			inQuotes = true
			quoteChar = c
			cur += string(c)
			continue
		}
		if c == ' ' || c == '\t' {
			if cur != "" {
				parts = append(parts, cur)
				cur = ""
			}
			continue
		}
		cur += string(c)
	}
	if cur != "" {
		parts = append(parts, cur)
	}

	for _, tok := range parts {
		if kv := strings.SplitN(tok, "=", 2); len(kv) == 2 {
			v := trimQuotes(kv[1])
			m[kv[0]] = v
		} else {
			m[tok] = "true"
		}
	}
	return m
}

// Parse accepts an AccumulatedResponse and returns any complete StreamMessage events found.
func (p *FenceParser) Parse(ctx context.Context, res *AccumulatedResponse) ([]*StreamMessage, error) {
	if res == nil || res.Chunk == nil {
		return nil, nil
	}
	out := []*StreamMessage{}
	p.buf.WriteString(res.Chunk.Message.Content)
	s := p.buf.String()

	for {
		start := strings.Index(s, "```diffstream")
		if start == -1 {
			break
		}
		// find end of opening line
		nl := strings.IndexByte(s[start:], '\n')
		if nl == -1 {
			// no newline yet, wait for more input
			break
		}
		header := strings.TrimSpace(s[start : start+nl])
		bodyStart := start + nl + 1
		// be permissive: accept closing fence even if there's no preceding newline
		endRel := strings.Index(s[bodyStart:], "```")
		if endRel == -1 {
			break
		}
		// compute absolute index
		endIdx := bodyStart + endRel
		body := s[bodyStart:endIdx]
		// if body begins with a newline (typical), trim the first one to preserve exact content
		if len(body) > 0 && body[0] == '\n' {
			body = body[1:]
		}
		nextPos := endIdx + len("```")

		attrs := parseHeaderAttrs(strings.TrimSpace(header[len("```diffstream"):]))
		sm := &StreamMessage{}
		if v, ok := attrs["type"]; ok {
			sm.Type = v
		}
		if v, ok := attrs["op"]; ok {
			sm.Op = v
		}
		if v, ok := attrs["path"]; ok {
			sm.Path = v
		}
		if v, ok := attrs["encoding"]; ok {
			sm.ContentEncoding = v
		}
		if v, ok := attrs["content_encoding"]; ok {
			sm.ContentEncoding = v
		}
		if v, ok := attrs["file_id"]; ok {
			sm.FileID = v
		}
		if v, ok := attrs["data_encoding"]; ok {
			sm.DataEncoding = v
		}
		if v, ok := attrs["chunk_index"]; ok {
			sm.ChunkIndex = atoiOrZero(v)
		}
		if v, ok := attrs["total_chunks"]; ok {
			sm.TotalChunks = atoiOrZero(v)
		}
		if v, ok := attrs["message"]; ok {
			sm.Message = v
		}

		// decide where to put body; preserve body exactness for utf-8, but
		// trim a single trailing newline if model authors included one.
		if sm.Type == "chunk" || sm.DataEncoding != "" {
			bodyTrim := strings.TrimSpace(body)
			sm.Data = bodyTrim
			if sm.DataEncoding == "" {
				sm.DataEncoding = "base64"
			}
		} else {
			// keep as-is but strip a single trailing newline common in model outputs
			if strings.HasSuffix(body, "\n") {
				body = strings.TrimSuffix(body, "\n")
			}
			sm.Content = body
			if sm.ContentEncoding == "" {
				sm.ContentEncoding = "utf-8"
			}
		}

		out = append(out, sm)
		s = s[nextPos:]
	}

	p.buf.Reset()
	p.buf.WriteString(s)
	return out, nil
}

// AttachMessageParserToAccumulator composes a generic MessageParser with a DiffParser.
// It feeds each AccumulatedResponse into the MessageParser; for every parsed
// StreamMessage it invokes diff.HandleMessage. The original AccumulatedResponse
// values are forwarded unchanged to the returned channel.
func AttachMessageParserToAccumulator(ctx context.Context, input <-chan *AccumulatedResponse, mp MessageParser, diff *DiffParser) <-chan *AccumulatedResponse {
	out := make(chan *AccumulatedResponse)
	go func() {
		defer close(out)
		for res := range input {
			if res != nil && mp != nil && diff != nil {
				msgs, err := mp.Parse(ctx, res)
				if err != nil {
					fmt.Printf("message parser error: %v\n", err)
				} else {
					for _, m := range msgs {
						if err := diff.HandleMessage(ctx, m); err != nil {
							fmt.Printf("diff handler error: %v\n", err)
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
