package ai

import (
	"context"
	"strings"
	"testing"
)

func TestFenceParser_SimpleGitDiffBlock(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	diff := strings.Join([]string{
		"```diff",
		"--- a/main.go",
		"+++ b/main.go",
		"@@ -1,3 +1,4 @@",
		" package main",
		"+// added comment",
		" func main() {}",
		"```",
	}, "\n")

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: diff},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}

	block := blocks[0]
	if block.Type != "diff" {
		t.Fatalf("expected type=diff, got %q", block.Type)
	}

	if block.Content != strings.Join([]string{
		"--- a/main.go",
		"+++ b/main.go",
		"@@ -1,3 +1,4 @@",
		" package main",
		"+// added comment",
		" func main() {}",
		"",
	}, "\n") {
		t.Fatalf("unexpected content:\n%q", block.Content)
	}
}

func TestFenceParser_ChunkedAcrossCalls_GitDiffBlock(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	part1 := "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-hello"
	part2 := "\n+world\n```"

	res1 := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: part1},
		},
	}
	blocks1, err := p.ParseBlocks(ctx, res1)
	if err != nil {
		t.Fatalf("parse1 error: %v", err)
	}
	if len(blocks1) != 0 {
		t.Fatalf("expected 0 blocks for partial input, got %d", len(blocks1))
	}

	res2 := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: part2},
		},
	}
	blocks2, err := p.ParseBlocks(ctx, res2)
	if err != nil {
		t.Fatalf("parse2 error: %v", err)
	}
	if len(blocks2) != 1 {
		t.Fatalf("expected 1 block after completion, got %d", len(blocks2))
	}

	block := blocks2[0]
	if block.Type != "diff" {
		t.Fatalf("expected type=diff, got %q", block.Type)
	}
	if block.Content != "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-hello\n+world\n" {
		t.Fatalf("unexpected block: %+v", block)
	}
}

func TestFenceParser_ExtractsAnyFenceType(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "```diffstream\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n```"},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block for diffstream fence, got %d", len(blocks))
	}
	if blocks[0].Type != "diffstream" {
		t.Fatalf("expected type=diffstream, got %q", blocks[0].Type)
	}
}

func TestFenceParser_MultipleBlocks(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	content := strings.Join([]string{
		"before",
		"```diff",
		"--- a/one.txt",
		"+++ b/one.txt",
		"@@ -1 +1 @@",
		"-old",
		"+new",
		"```",
		"middle",
		"```diff",
		"--- a/two.txt",
		"+++ b/two.txt",
		"@@ -0,0 +1 @@",
		"+created",
		"```",
		"after",
	}, "\n")

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: content},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}

	if blocks[0].Type != "diff" || blocks[1].Type != "diff" {
		t.Fatalf("expected both blocks to be diff, got %+v", blocks)
	}

	if blocks[0].Content != "--- a/one.txt\n+++ b/one.txt\n@@ -1 +1 @@\n-old\n+new\n" {
		t.Fatalf("unexpected first block content: %q", blocks[0].Content)
	}
	if blocks[1].Content != "--- a/two.txt\n+++ b/two.txt\n@@ -0,0 +1 @@\n+created\n" {
		t.Fatalf("unexpected second block content: %q", blocks[1].Content)
	}
}

func TestFenceParser_ExtractsJsonFenceType(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "```json\n{\"hello\":\"world\"}\n```"},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block for json fence type, got %d", len(blocks))
	}
	if blocks[0].Type != "json" {
		t.Fatalf("expected type=json, got %q", blocks[0].Type)
	}
}

func TestFenceParser_IgnoresFenceHeaderWithExtraText(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "```diff extra\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n```"},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 0 {
		t.Fatalf("expected 0 blocks for non-exact diff header, got %d", len(blocks))
	}
}

func TestFenceParser_PreservesLeadingBlankLineInBody(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "```diff\n\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n```"},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	if blocks[0].Content != "\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n" {
		t.Fatalf("unexpected content with leading blank line: %q", blocks[0].Content)
	}
}

func TestFenceParser_RetainsIncompleteFenceUntilComplete(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res1 := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new"},
		},
	}

	blocks1, err := p.ParseBlocks(ctx, res1)
	if err != nil {
		t.Fatalf("parse1 error: %v", err)
	}
	if len(blocks1) != 0 {
		t.Fatalf("expected 0 blocks for incomplete fence, got %d", len(blocks1))
	}

	res2 := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "\n```"},
		},
	}

	blocks2, err := p.ParseBlocks(ctx, res2)
	if err != nil {
		t.Fatalf("parse2 error: %v", err)
	}
	if len(blocks2) != 1 {
		t.Fatalf("expected 1 block after completing fence, got %d", len(blocks2))
	}
	if blocks2[0].Type != "diff" {
		t.Fatalf("expected type=diff, got %q", blocks2[0].Type)
	}
}

func TestFenceParser_IgnoresProseBeforeAndAfterFence(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: "hello\nthere\n```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n```\nbye"},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	if blocks[0].Content != "--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n" {
		t.Fatalf("unexpected parsed content: %q", blocks[0].Content)
	}
}

func TestFenceParser_BackticksInsideBlockContent(t *testing.T) {
	p := NewFenceParser()
	ctx := context.Background()

	content := strings.Join([]string{
		"```diff",
		"--- a/README.md",
		"+++ b/README.md",
		"@@ -1,3 +1,3 @@",
		" Here is code:",
		"-```go",
		"+```js",
		" fmt.Println(\"hi\")",
		"```",
	}, "\n")

	res := &AccumulatedResponse{
		Chunk: &ChatResponse{
			Message: Message{Content: content},
		},
	}

	blocks, err := p.ParseBlocks(ctx, res)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}

	// The content should include everything between the fences,
	// including the inner backticks because they are not at the start of the line alone.
	expectedContent := "--- a/README.md\n+++ b/README.md\n@@ -1,3 +1,3 @@\n Here is code:\n-```go\n+```js\n fmt.Println(\"hi\")\n"
	if blocks[0].Content != expectedContent {
		t.Fatalf("unexpected content:\n%q\nexpected:\n%q", blocks[0].Content, expectedContent)
	}
}
