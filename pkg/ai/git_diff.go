package ai

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/sergi/go-diff/diffmatchpatch"
)

// DiffOperation represents a single unified diff application request.
type DiffOperation struct {
	Content string
}

// OperationResult represents the outcome of a single diff-related operation.
type OperationResult struct {
	Success bool                   `json:"success"`
	Op      string                 `json:"op"`
	Path    string                 `json:"path,omitempty"`
	Message string                 `json:"message,omitempty"`
	Extra   map[string]interface{} `json:"extra,omitempty"`
}

// OperationHandler is a composable handler interface that receives repoRoot and
// a unified diff operation. Implementations should return an OperationResult
// describing what succeeded or failed.
type OperationHandler interface {
	HandleDiff(ctx context.Context, repoRoot string, op *DiffOperation) OperationResult
	HandleCommit(ctx context.Context, repoRoot string, message string) OperationResult
}

// DiffParser applies unified git diffs to a repository root and records
// operation reports for later inspection.
type DiffParser struct {
	repoRoot string
	mu       sync.Mutex

	handler OperationHandler
	reports []OperationResult
}

// NewDiffParser creates a parser that applies unified diffs within repoRoot.
func NewDiffParser(repoRoot string) *DiffParser {
	return &DiffParser{
		repoRoot: repoRoot,
	}
}

// DefaultOperationHandler applies unified git diffs and performs git commits.
type DefaultOperationHandler struct{}

func NewDiffOperationHandler() OperationHandler {
	return &DefaultOperationHandler{}
}

func (d *DefaultOperationHandler) HandleDiff(ctx context.Context, repoRoot string, op *DiffOperation) OperationResult {
	if op == nil {
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: "nil diff operation",
		}
	}
	if strings.TrimSpace(op.Content) == "" {
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: "empty diff content",
		}
	}

	tmpPatch, err := os.CreateTemp("", "streamed-diff-*.patch")
	if err != nil {
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: err.Error(),
		}
	}
	defer os.Remove(tmpPatch.Name())

	if _, err := tmpPatch.WriteString(op.Content); err != nil {
		tmpPatch.Close()
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: err.Error(),
		}
	}
	if err := tmpPatch.Close(); err != nil {
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: err.Error(),
		}
	}

	cmd := exec.CommandContext(ctx, "git", "apply", "--whitespace=fix", tmpPatch.Name())
	cmd.Dir = repoRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("[diff] git apply failed: %v: %s\n", err, string(out))
		fmt.Printf("[diff] attempting fuzzy apply fallback for content:\n%s\n", op.Content)
		return d.handleFuzzyApply(ctx, repoRoot, op.Content)
	}

	fmt.Printf("[diff] applied patch in %s\n", repoRoot)
	return OperationResult{
		Success: true,
		Op:      "diff",
	}
}

func (d *DefaultOperationHandler) handleFuzzyApply(ctx context.Context, repoRoot string, content string) OperationResult {
	// Step 1: Split content into per-file chunks
	// We look for patterns like:
	// --- a/path
	// +++ b/path
	// ...hunk...
	// We handle optional leading spaces and both \n and \r\n
	reFile := regexp.MustCompile(`(?m)^[ \t]*--- (?:a/)?([^\s\t\n\r]+)[\r]*\n[ \t]*\+\+\+ (?:b/)?([^\s\t\n\r]+)`)
	indices := reFile.FindAllStringIndex(content, -1)

	if len(indices) == 0 {
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: "fuzzy apply: could not find any file headers with --- and +++",
		}
	}

	type fileHunk struct {
		path string
		hunk string
	}
	var hunks []fileHunk

	for i := 0; i < len(indices); i++ {
		start := indices[i][0]
		end := len(content)
		if i+1 < len(indices) {
			end = indices[i+1][0]
		}

		// Extract path from the --- header
		submatches := reFile.FindStringSubmatch(content[start:indices[i][1]])
		path := submatches[1]
		if path == "/dev/null" {
			path = submatches[2]
		}
		if path == "/dev/null" {
			continue // Should not happen for both
		}

		hunks = append(hunks, fileHunk{
			path: path,
			hunk: content[start:end],
		})
	}

	dmp := diffmatchpatch.New()
	var successes []string
	var failures []string

	for _, fh := range hunks {
		absPath := filepath.Join(repoRoot, fh.path)
		original, err := os.ReadFile(absPath)
		if err != nil {
			if os.IsNotExist(err) {
				// We only allow creating a file if it looks like a creation diff 
				// (using /dev/null or if it was explicitly requested).
				// For fuzzy matching, we'll be conservative.
				if !strings.Contains(fh.hunk, "/dev/null") {
					failures = append(failures, fmt.Sprintf("%s (file does not exist and no /dev/null header found)", fh.path))
					continue
				}
				original = []byte("")
			} else {
				failures = append(failures, fmt.Sprintf("%s (read error: %v)", fh.path, err))
				continue
			}
		}

		// Prepare patch text: go-diff expects @@ headers.
		// If missing, we prepend a dummy one.
		// Filter to keep only diff lines (@@, +, -, or space)
		// and strip git headers that go-diff might not like.
		lines := strings.Split(strings.TrimRight(fh.hunk, "\n"), "\n")
		var filtered []string
		hasHeader := false
		for _, line := range lines {
			// Trim any trailing \r
			line = strings.TrimRight(line, "\r")
			
			trimmed := strings.TrimLeft(line, " \t")
			// Skip file headers (--- and +++)
			if strings.HasPrefix(trimmed, "---") || strings.HasPrefix(trimmed, "+++") {
				continue
			}
			
			if strings.HasPrefix(trimmed, "@@") {
				hasHeader = true
				filtered = append(filtered, line) // Keep original line for @@
			} else if strings.HasPrefix(trimmed, "+") ||
				strings.HasPrefix(trimmed, "-") ||
				strings.HasPrefix(line, " ") ||
				line == "" { 
				filtered = append(filtered, line)
			}
		}

		patchText := strings.Join(filtered, "\n")
		if !hasHeader {
			patchText = "@@ -0,0 +0,0 @@\n" + patchText
		}

		patches, err := dmp.PatchFromText(patchText)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s (parse error: %v)", fh.path, err))
			continue
		}

		applied, results := dmp.PatchApply(patches, string(original))
		hunkSuccess := true
		for _, r := range results {
			if !r {
				hunkSuccess = false
				break
			}
		}

		if !hunkSuccess {
			var hunkContext string
			if len(patches) > 0 {
				hunkContext = fmt.Sprintf(" (first hunk: %s)", patches[0].String())
			}
			failures = append(failures, fmt.Sprintf("%s (fuzzy match failed - tip: read the file first to ensure context is correct)%s", fh.path, hunkContext))
			continue
		}

		if err := os.MkdirAll(filepath.Dir(absPath), 0755); err != nil {
			failures = append(failures, fmt.Sprintf("%s (mkdir error: %v)", fh.path, err))
			continue
		}
		if err := os.WriteFile(absPath, []byte(applied), 0644); err != nil {
			failures = append(failures, fmt.Sprintf("%s (write error: %v)", fh.path, err))
			continue
		}
		successes = append(successes, fh.path)
	}

	if len(failures) > 0 {
		msg := "fuzzy apply failed: " + strings.Join(failures, "; ")
		if len(successes) > 0 {
			msg += " (but succeeded for: " + strings.Join(successes, ", ") + ")"
		}
		return OperationResult{
			Success: false,
			Op:      "diff",
			Message: msg,
		}
	}

	fmt.Printf("[diff] Applied fuzzy patch to: %s\n", strings.Join(successes, ", "))
	return OperationResult{
		Success: true,
		Op:      "diff",
		Message: "applied via fuzzy fallback: " + strings.Join(successes, ", "),
	}
}

func (d *DefaultOperationHandler) HandleCommit(ctx context.Context, repoRoot string, message string) OperationResult {
	msg := strings.TrimSpace(message)
	if msg == "" {
		msg = "stream-commit"
	}

	cmd := exec.CommandContext(ctx, "git", "add", "-A")
	cmd.Dir = repoRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		return OperationResult{
			Success: false,
			Op:      "commit",
			Message: fmt.Sprintf("git add failed: %v: %s", err, string(out)),
		}
	}

	cmd = exec.CommandContext(ctx, "git", "commit", "-m", msg)
	cmd.Dir = repoRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		if bytes.Contains(out, []byte("nothing to commit")) {
			return OperationResult{
				Success: true,
				Op:      "commit",
				Message: "nothing to commit",
			}
		}
		return OperationResult{
			Success: false,
			Op:      "commit",
			Message: fmt.Sprintf("git commit failed: %v: %s", err, string(out)),
		}
	}

	fmt.Printf("[diff] committed in %s: %s\n", repoRoot, msg)
	return OperationResult{
		Success: true,
		Op:      "commit",
		Message: msg,
	}
}

// SetHandler attaches an OperationHandler to the parser.
func (p *DiffParser) SetHandler(h OperationHandler) {
	p.handler = h
}

// PopReports returns and clears any reports produced by the handler.
func (p *DiffParser) PopReports() []OperationResult {
	p.mu.Lock()
	defer p.mu.Unlock()

	reports := p.reports
	p.reports = nil
	return reports
}

func (p *DiffParser) appendReport(r OperationResult) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reports = append(p.reports, r)
}

// ApplyDiff applies a unified diff operation and records the result.
func (p *DiffParser) ApplyDiff(ctx context.Context, op *DiffOperation) error {
	if p.handler != nil {
		rep := p.handler.HandleDiff(ctx, p.repoRoot, op)
		p.appendReport(rep)
		if !rep.Success {
			return fmt.Errorf("%s", rep.Message)
		}
		return nil
	}

	rep := (&DefaultOperationHandler{}).HandleDiff(ctx, p.repoRoot, op)
	p.appendReport(rep)
	if !rep.Success {
		return fmt.Errorf("%s", rep.Message)
	}
	return nil
}

// ApplyBlock applies a fenced streamed block when it is of type diff.
func (p *DiffParser) ApplyBlock(ctx context.Context, block *StreamedBlock) error {
	if block == nil {
		return nil
	}
	if block.Type != "diff" {
		return nil
	}
	if !block.Done {
		return nil
	}
	return p.ApplyDiff(ctx, &DiffOperation{Content: block.Content})
}

// Commit stages and commits repository changes using the configured handler.
func (p *DiffParser) Commit(ctx context.Context, message string) error {
	if p.handler != nil {
		rep := p.handler.HandleCommit(ctx, p.repoRoot, message)
		p.appendReport(rep)
		if !rep.Success {
			return fmt.Errorf("%s", rep.Message)
		}
		return nil
	}

	rep := (&DefaultOperationHandler{}).HandleCommit(ctx, p.repoRoot, message)
	p.appendReport(rep)
	if !rep.Success {
		return fmt.Errorf("%s", rep.Message)
	}
	return nil
}

// GitDiffBlockHandler applies fenced diff blocks through a DiffParser.
type GitDiffBlockHandler struct {
	diff *DiffParser
}

func NewGitDiffBlockHandler(diff *DiffParser) *GitDiffBlockHandler {
	return &GitDiffBlockHandler{diff: diff}
}

func (h *GitDiffBlockHandler) HandleBlock(ctx context.Context, block *StreamedBlock) error {
	if h == nil || h.diff == nil {
		return nil
	}
	return h.diff.ApplyBlock(ctx, block)
}

// RepoRoot returns the parser's repository root.
func (p *DiffParser) RepoRoot() string {
	return p.repoRoot
}

// AbsPathFor resolves a relative path inside repoRoot safely.
func (p *DiffParser) AbsPathFor(rel string) (string, error) {
	rel = filepath.ToSlash(rel)
	clean := filepath.Clean(rel)
	if filepath.IsAbs(clean) {
		return "", fmt.Errorf("absolute paths not allowed: %q", rel)
	}
	if strings.HasPrefix(clean, ".."+string(filepath.Separator)) || clean == ".." {
		return "", fmt.Errorf("path traversal detected: %q", rel)
	}
	return filepath.Join(p.repoRoot, clean), nil
}
