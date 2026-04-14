package ai

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestGitDiffBlockHandler_AppliesUnifiedDiff(t *testing.T) {
	tmp := t.TempDir()

	if out, err := exec.Command("git", "-C", tmp, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "config", "user.name", "Test").CombinedOutput(); err != nil {
		t.Fatalf("git config user.name failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "config", "user.email", "test@example.com").CombinedOutput(); err != nil {
		t.Fatalf("git config user.email failed: %v: %s", err, string(out))
	}

	mainPath := filepath.Join(tmp, "main.go")
	original := "package main\n\nfunc add(a int, b int) int {\n\treturn a + b\n}\n"
	if err := os.WriteFile(mainPath, []byte(original), 0o644); err != nil {
		t.Fatalf("write seed file failed: %v", err)
	}
	if out, err := exec.Command("git", "-C", tmp, "add", "main.go").CombinedOutput(); err != nil {
		t.Fatalf("git add failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "commit", "-m", "seed").CombinedOutput(); err != nil {
		t.Fatalf("git commit failed: %v: %s", err, string(out))
	}

	dp := NewDiffParser(tmp)
	handler := NewGitDiffBlockHandler(dp)

	block := &StreamedBlock{
		Type: "diff",
		Done: true,
		Content: strings.Join([]string{
			"--- a/main.go",
			"+++ b/main.go",
			"@@ -1,5 +1,6 @@",
			" package main",
			" ",
			" func add(a int, b int) int {",
			"+\t// Computes the sum of two integer arguments",
			" \treturn a + b",
			" }",
			"",
		}, "\n"),
	}

	if err := handler.HandleBlock(context.Background(), block); err != nil {
		t.Fatalf("HandleBlock failed: %v", err)
	}

	got, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatalf("reading patched file failed: %v", err)
	}

	wantFragment := "\t// Computes the sum of two integer arguments\n\treturn a + b\n"
	if !strings.Contains(string(got), wantFragment) {
		t.Fatalf("patched file missing fragment %q in %q", wantFragment, string(got))
	}
}

func TestGitDiffBlockHandler_IgnoresNonDiffBlocks(t *testing.T) {
	tmp := t.TempDir()
	dp := NewDiffParser(tmp)
	handler := NewGitDiffBlockHandler(dp)

	block := &StreamedBlock{
		Type:    "json",
		Content: "{\"hello\":\"world\"}\n",
	}

	if err := handler.HandleBlock(context.Background(), block); err != nil {
		t.Fatalf("expected non-diff block to be ignored, got error: %v", err)
	}
}

func TestGitDiffBlockHandler_FailsForInvalidPatch(t *testing.T) {
	tmp := t.TempDir()

	if out, err := exec.Command("git", "-C", tmp, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v: %s", err, string(out))
	}

	dp := NewDiffParser(tmp)
	handler := NewGitDiffBlockHandler(dp)

	block := &StreamedBlock{
		Type: "diff",
		Done: true,
		Content: strings.Join([]string{
			"--- a/does-not-exist.go",
			"+++ b/does-not-exist.go",
			"@@ -1 +1 @@",
			"-old",
			"+new",
			"",
		}, "\n"),
	}

	err := handler.HandleBlock(context.Background(), block)
	if err == nil {
		t.Fatalf("expected invalid patch to fail")
	}
	if !strings.Contains(err.Error(), "fuzzy apply failed:") {
		t.Fatalf("expected fuzzy apply failure for missing file, got: %v", err)
	}
}

func TestGitDiffBlockHandler_CreatesNewFile(t *testing.T) {
	tmp := t.TempDir()

	if out, err := exec.Command("git", "-C", tmp, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v: %s", err, string(out))
	}

	dp := NewDiffParser(tmp)
	handler := NewGitDiffBlockHandler(dp)

	block := &StreamedBlock{
		Type: "diff",
		Done: true,
		Content: strings.Join([]string{
			"diff --git a/docs/example.txt b/docs/example.txt",
			"new file mode 100644",
			"--- /dev/null",
			"+++ b/docs/example.txt",
			"@@ -0,0 +1,3 @@",
			"+first line",
			"+second line",
			"+third line",
			"",
		}, "\n"),
	}

	if err := handler.HandleBlock(context.Background(), block); err != nil {
		t.Fatalf("HandleBlock create file failed: %v", err)
	}

	got, err := os.ReadFile(filepath.Join(tmp, "docs", "example.txt"))
	if err != nil {
		t.Fatalf("expected created file: %v", err)
	}
	if string(got) != "first line\nsecond line\nthird line\n" {
		t.Fatalf("unexpected created file content: %q", string(got))
	}
}

func TestGitDiffBlockHandler_ProducesReportThroughDiffParser(t *testing.T) {
	tmp := t.TempDir()

	if out, err := exec.Command("git", "-C", tmp, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "config", "user.name", "Test").CombinedOutput(); err != nil {
		t.Fatalf("git config user.name failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "config", "user.email", "test@example.com").CombinedOutput(); err != nil {
		t.Fatalf("git config user.email failed: %v: %s", err, string(out))
	}

	mainPath := filepath.Join(tmp, "main.go")
	original := "package main\n\nfunc main() {}\n"
	if err := os.WriteFile(mainPath, []byte(original), 0o644); err != nil {
		t.Fatalf("write seed file failed: %v", err)
	}
	if out, err := exec.Command("git", "-C", tmp, "add", "main.go").CombinedOutput(); err != nil {
		t.Fatalf("git add failed: %v: %s", err, string(out))
	}
	if out, err := exec.Command("git", "-C", tmp, "commit", "-m", "seed").CombinedOutput(); err != nil {
		t.Fatalf("git commit failed: %v: %s", err, string(out))
	}

	dp := NewDiffParser(tmp)
	dp.SetHandler(&DefaultOperationHandler{})
	handler := NewGitDiffBlockHandler(dp)

	block := &StreamedBlock{
		Type: "diff",
		Done: true,
		Content: strings.Join([]string{
			"--- a/main.go",
			"+++ b/main.go",
			"@@ -1,3 +1,4 @@",
			" package main",
			" ",
			"+// hello",
			" func main() {}",
			"",
		}, "\n"),
	}

	if err := handler.HandleBlock(context.Background(), block); err != nil {
		t.Fatalf("HandleBlock failed: %v", err)
	}

	reports := dp.PopReports()
	if len(reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(reports))
	}
	if !reports[0].Success {
		t.Fatalf("expected success report, got %+v", reports[0])
	}
	if reports[0].Op != "diff" {
		t.Fatalf("expected diff report, got %+v", reports[0])
	}
}

func TestDefaultOperationHandler_FuzzyFallback(t *testing.T) {
	tmp := t.TempDir()

	mainPath := filepath.Join(tmp, "main.go")
	original := "package main\n\nfunc main() {\n\t// old logic\n}\n"
	if err := os.WriteFile(mainPath, []byte(original), 0o644); err != nil {
		t.Fatalf("write seed file failed: %v", err)
	}

	handler := &DefaultOperationHandler{}

	// Malformed diff (no @@, context changed slightly but fuzzy match should find it)
	block := &DiffOperation{
		Content: strings.Join([]string{
			"--- a/main.go",
			"+++ b/main.go",
			"- \t// old logic",
			"+ \t// new logic",
		}, "\n"),
	}

	// git apply will fail because of missing @@ and possibly other malformations
	res := handler.HandleDiff(context.Background(), tmp, block)
	if !res.Success {
		t.Fatalf("Fuzzy apply failed: %s", res.Message)
	}
	if !strings.Contains(res.Message, "applied via fuzzy fallback") {
		t.Fatalf("Expected fuzzy fallback message, got: %s", res.Message)
	}

	got, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatalf("reading patched file failed: %v", err)
	}

	if !strings.Contains(string(got), "// new logic") {
		t.Fatalf("patched file missing new logic in %q", string(got))
	}
}

func TestGitDiffBlockHandler_CreatesDirectoriesAutomatically(t *testing.T) {
	tmp := t.TempDir()

	if out, err := exec.Command("git", "-C", tmp, "init").CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v: %s", err, string(out))
	}

	dp := NewDiffParser(tmp)
	handler := NewGitDiffBlockHandler(dp)

	// Apply diff to a file in a non-existent subdirectory
	block := &StreamedBlock{
		Type: "diff",
		Done: true,
		Content: strings.Join([]string{
			"--- a/newdir/nested/file.txt",
			"+++ b/newdir/nested/file.txt",
			"@@ -0,0 +1 @@",
			"+nested content",
			"",
		}, "\n"),
	}

	if err := handler.HandleBlock(context.Background(), block); err != nil {
		t.Fatalf("HandleBlock failed: %v", err)
	}

	got, err := os.ReadFile(filepath.Join(tmp, "newdir", "nested", "file.txt"))
	if err != nil {
		t.Fatalf("reading patched file failed: %v", err)
	}

	if string(got) != "nested content\n" {
		t.Fatalf("unexpected content: %q", string(got))
	}
}

func TestDefaultOperationHandler_FuzzyApplyRobustness(t *testing.T) {
	tmp := t.TempDir()

	mainPath := filepath.Join(tmp, "main.go")
	original := "package main\r\n\r\nfunc main() {\r\n\t// first line\r\n\t// second line\r\n}\r\n"
	if err := os.WriteFile(mainPath, []byte(original), 0o644); err != nil {
		t.Fatalf("write seed file failed: %v", err)
	}

	handler := &DefaultOperationHandler{}

	// Test Case 1: CRLF line endings in diff + Indented headers + Missing context spaces
	// The diff is "lazy" (no @@) and has \r\n explicitly
	content := "   --- a/main.go\r\n   +++ b/main.go\r\n" +
		" \t// first line\r\n" +
		"\r\n" + // Empty line with NO space (missing context space)
		"- \t// second line\r\n" +
		"+ \t// patched line\r\n"

	block := &DiffOperation{
		Content: content,
	}

	res := handler.HandleDiff(context.Background(), tmp, block)
	if !res.Success {
		t.Fatalf("Fuzzy apply failed: %s", res.Message)
	}

	got, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatalf("reading patched file failed: %v", err)
	}

	// The fuzzy matcher might have changed \r\n to \n depending on implementation, 
	// but the content should be correct.
	gotStr := strings.ReplaceAll(string(got), "\r\n", "\n")
	if !strings.Contains(gotStr, "// patched line") {
		t.Fatalf("patched file missing new logic in %q", string(got))
	}
}

