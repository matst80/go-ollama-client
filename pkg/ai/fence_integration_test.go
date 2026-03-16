package ai

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Integration test: feed a fenced diffstream block via AttachMessageParserToAccumulator
// and verify DefaultOperationHandler wrote the file and committed when commit op is sent.
func TestFenceIntegration_WriteAndCommit(t *testing.T) {
	tmp := t.TempDir()
	repo := tmp
	// init git repo in tmp and set basic config so commits succeed
	if out, err := exec.Command("git", "-C", repo, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init in tmp failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", repo, "config", "user.name", "Test").CombinedOutput(); err != nil {
		t.Fatalf("git config user.name failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", repo, "config", "user.email", "test@example.com").CombinedOutput(); err != nil {
		t.Fatalf("git config user.email failed: %v: %s", err, string(out))
	}

	// create parser and handler
	dp := NewDiffParser(repo)
	dp.SetHandler(&DefaultOperationHandler{})

	// build fenced message that adds a file
	content := "hello world\n"
	fence := "```diffstream type=file op=add path=info.txt encoding=utf-8\n" + content + "\n```"
	acc := make(chan *AccumulatedResponse, 2)
	out := AttachMessageParserToAccumulator(context.Background(), acc, NewFenceParser(), dp)

	// send the accumulated response
	acc <- &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: fence}}}
	close(acc)

	// drain out channel
	for range out {
	}

	// verify file exists
	got, err := os.ReadFile(filepath.Join(repo, "info.txt"))
	if err != nil {
		t.Fatalf("expected file written: %v", err)
	}
	if string(got) != content+"\n" && string(got) != content {
		t.Fatalf("content mismatch: %q", string(got))
	}

	// now commit via a commit message fence
	acc2 := make(chan *AccumulatedResponse, 2)
	out2 := AttachMessageParserToAccumulator(context.Background(), acc2, NewFenceParser(), dp)
	acc2 <- &AccumulatedResponse{Chunk: &ChatResponse{Message: Message{Content: "```diffstream type=commit message=\"test commit\" finalize=true\n\n```"}}}
	close(acc2)
	for range out2 {
	}

	// Confirm that the OperationHandler produced a commit report
	reports := dp.PopReports()
	found := false
	for _, r := range reports {
		if r.Op == "commit" {
			found = true
			if !r.Success {
				t.Fatalf("commit reported failure: %s", r.Message)
			}
		}
	}
	if !found {
		t.Fatalf("expected commit report, none found; reports=%+v", reports)
	}
}
