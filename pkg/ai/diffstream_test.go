package ai

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestDiffParser_FileWriteDelete(t *testing.T) {
	tmp := t.TempDir()
	dp := NewDiffParser(tmp)
	ctx := context.Background()

	content := "hello\n"
	h := sha256.Sum256([]byte(content))
	nd := fmt.Sprintf(`{"type":"file","op":"add","path":"sub/hello.txt","content_encoding":"utf-8","content":%q,"sha256":"%s","atomic":true}`,
		content, hex.EncodeToString(h[:]))

	res := &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: nd}}}
	if err := dp.ProcessChunk(ctx, res); err != nil {
		t.Fatalf("ProcessChunk add failed: %v", err)
	}

	got, err := os.ReadFile(filepath.Join(tmp, "sub", "hello.txt"))
	if err != nil {
		t.Fatalf("expected file written: %v", err)
	}
	if string(got) != content {
		t.Fatalf("file content mismatch: got %q want %q", string(got), content)
	}

	// delete
	ndDel := `{"type":"file","op":"delete","path":"sub/hello.txt"}`
	res2 := &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: ndDel}}}
	if err := dp.ProcessChunk(ctx, res2); err != nil {
		t.Fatalf("ProcessChunk delete failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmp, "sub", "hello.txt")); !os.IsNotExist(err) {
		t.Fatalf("expected file removed, stat err=%v", err)
	}
}

func TestDiffParser_ChunkReassembly(t *testing.T) {
	tmp := t.TempDir()
	dp := NewDiffParser(tmp)
	ctx := context.Background()

	part1 := "hello "
	part2 := "world"
	c1 := base64.StdEncoding.EncodeToString([]byte(part1))
	c2 := base64.StdEncoding.EncodeToString([]byte(part2))

	chunk1 := fmt.Sprintf(`{"type":"chunk","file_id":"f1","chunk_index":0,"total_chunks":2,"data_encoding":"base64","data":"%s"}`, c1)
	chunk2 := fmt.Sprintf(`{"type":"chunk","file_id":"f1","chunk_index":1,"total_chunks":2,"data_encoding":"base64","data":"%s"}`, c2)

	if err := dp.ProcessChunk(ctx, &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: chunk1}}}); err != nil {
		t.Fatalf("chunk1 failed: %v", err)
	}
	if err := dp.ProcessChunk(ctx, &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: chunk2}}}); err != nil {
		t.Fatalf("chunk2 failed: %v", err)
	}

	outPath := filepath.Join(tmp, ".stream", "f1")
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("assembled file missing: %v", err)
	}
	if string(data) != part1+part2 {
		t.Fatalf("assembled content mismatch: got %q want %q", string(data), part1+part2)
	}
}

// rec implements OperationHandler for tests and records OperationResult entries.
type rec struct{ r []OperationResult }

func (r *rec) HandleFile(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	rep := OperationResult{Success: true, Op: "file", Path: m.Path}
	r.r = append(r.r, rep)
	return rep
}

func (r *rec) HandleChunk(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	rep := OperationResult{Success: true, Op: "chunk", Extra: map[string]interface{}{"file_id": m.FileID}}
	r.r = append(r.r, rep)
	return rep
}

func (r *rec) HandleCommit(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	rep := OperationResult{Success: true, Op: "commit", Message: m.Message}
	r.r = append(r.r, rep)
	return rep
}

func TestDiffParser_HandlerReports(t *testing.T) {
	tmp := t.TempDir()
	dp := NewDiffParser(tmp)
	ctx := context.Background()

	// simple handler that records results
	var rh rec
	dp.SetHandler(&rh)

	nd := `{"type":"file","op":"add","path":"a.txt","content_encoding":"utf-8","content":"x"}`
	if err := dp.ProcessChunk(ctx, &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: nd}}}); err != nil {
		t.Fatalf("ProcessChunk with handler failed: %v", err)
	}

	// PopReports should reflect handler's recorded results
	reps := dp.PopReports()
	if len(reps) != 1 {
		t.Fatalf("expected 1 report, got %d", len(reps))
	}
	if !reps[0].Success || reps[0].Op != "file" || reps[0].Path != "a.txt" {
		t.Fatalf("unexpected report: %+v", reps[0])
	}
}
