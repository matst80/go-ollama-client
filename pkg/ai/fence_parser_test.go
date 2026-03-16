package ai

import (
	"context"
	"testing"
)

func TestFenceParser_Simple(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()
	res := &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: "```diffstream type=file op=add path=sub/hello.txt encoding=utf-8\nhello world\n```"}}}
	msgs, err := p.Parse(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m := msgs[0]
	if m.Type != "file" || m.Op != "add" || m.Path != "sub/hello.txt" {
		t.Fatalf("unexpected message: %+v", m)
	}
	if m.Content != "hello world" {
		t.Fatalf("unexpected content: %q", m.Content)
	}
}

func TestFenceParser_ChunkedAcrossCalls(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()
	part1 := "```diffstream type=file op=add path=a.txt encoding=utf-8\nhello"
	part2 := " world\n```"
	res1 := &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: part1}}}
	msgs, err := p.Parse(ctx, res1)
	if err != nil {
		t.Fatalf("parse1 error: %v", err)
	}
	if len(msgs) != 0 {
		t.Fatalf("expected 0 messages for partial, got %d", len(msgs))
	}
	res2 := &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: part2}}}
	msgs2, err := p.Parse(ctx, res2)
	if err != nil {
		t.Fatalf("parse2 error: %v", err)
	}
	if len(msgs2) != 1 {
		t.Fatalf("expected 1 message after completion, got %d", len(msgs2))
	}
}
