package ai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// StreamMessage is a generic NDJSON record emitted by the model.
type StreamMessage struct {
	Type            string `json:"type"`
	ID              string `json:"id,omitempty"`
	Seq             int    `json:"seq,omitempty"`
	Op              string `json:"op,omitempty"`   // add/update/patch/delete/rename
	Path            string `json:"path,omitempty"` // relative POSIX path
	Mode            string `json:"mode,omitempty"`
	ContentEncoding string `json:"content_encoding,omitempty"` // "utf-8" | "base64" | "unified"
	Content         string `json:"content,omitempty"`          // raw, base64, or unified diff
	Sha256          string `json:"sha256,omitempty"`
	Atomic          bool   `json:"atomic,omitempty"`

	// chunk-style
	FileID       string `json:"file_id,omitempty"`
	ChunkIndex   int    `json:"chunk_index,omitempty"`
	TotalChunks  int    `json:"total_chunks,omitempty"`
	DataEncoding string `json:"data_encoding,omitempty"` // "base64"
	Data         string `json:"data,omitempty"`

	// commit/done
	Message  string `json:"message,omitempty"`
	Finalize bool   `json:"finalize,omitempty"`
}

// Handler types allow embedding custom behaviour when an operation is observed.
type FileHandler func(ctx context.Context, m *StreamMessage) error

// StreamChunkHandler avoids name collision with ChunkReader. It handles "chunk" events.
type StreamChunkHandler func(ctx context.Context, m *StreamMessage) error
type CommitHandler func(ctx context.Context, m *StreamMessage) error

// DiffParser applies streamed diff events to a repository root. It supports
// registering handlers for file/chunk/commit operations. If a handler is not
// registered, a sensible default (disk write/git apply) is used.
type DiffParser struct {
	repoRoot string
	mu       sync.Mutex

	// reassembly storage: fileID -> map[index]chunkBytes
	chunks map[string]map[int][]byte
	// store expected total for fileID
	totals map[string]int

	// optional handlers
	fileHandler   FileHandler
	chunkHandler  StreamChunkHandler
	commitHandler CommitHandler
	// pluggable operation handler
	handler OperationHandler
	// accumulated reports from handler; PopReports clears
	reports []OperationResult
}

// NewDiffParser creates a parser that writes into repoRoot.
func NewDiffParser(repoRoot string) *DiffParser {
	return &DiffParser{
		repoRoot: repoRoot,
		chunks:   make(map[string]map[int][]byte),
		totals:   make(map[string]int),
	}
}

// OperationResult represents the outcome of a single operation handled by the OperationHandler.
type OperationResult struct {
	Success bool                   `json:"success"`
	Op      string                 `json:"op"`
	Path    string                 `json:"path,omitempty"`
	Message string                 `json:"message,omitempty"`
	Extra   map[string]interface{} `json:"extra,omitempty"`
}

// OperationHandler is a composable handler interface that receives repoRoot and the parsed StreamMessage.
// Implementations should return an OperationResult describing what succeeded or failed.
type OperationHandler interface {
	HandleFile(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult
	HandleChunk(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult
	HandleCommit(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult
}

// DefaultOperationHandler implements OperationHandler by delegating to DiffParser's
// default behaviors (disk writes / chunk reassembly / git commit) and returns OperationResult.
type DefaultOperationHandler struct{}

func (d *DefaultOperationHandler) HandleFile(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	// Inline the default behaviour so we operate on the provided repoRoot
	// and emit visible logs for debugging.
	abs := filepath.Join(repoRoot, filepath.Clean(m.Path))
	// prevent traversal
	if strings.HasPrefix(m.Path, "../") || filepath.IsAbs(m.Path) {
		return OperationResult{Success: false, Op: "file", Path: m.Path, Message: "path not allowed"}
	}

	if m.Op == "delete" {
		if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
		}
		fmt.Printf("[diff] deleted %s\n", abs)
		return OperationResult{Success: true, Op: "file", Path: m.Path}
	}

	var data []byte
	switch m.ContentEncoding {
	case "base64":
		d, err := base64.StdEncoding.DecodeString(m.Content)
		if err != nil {
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
		}
		data = d
	case "utf-8", "":
		data = []byte(m.Content)
	case "unified":
		// write patch and run git apply in repoRoot
		tmpPatch, err := os.CreateTemp("", "stream-*.patch")
		if err != nil {
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
		}
		defer os.Remove(tmpPatch.Name())
		if _, err := tmpPatch.WriteString(m.Content); err != nil {
			tmpPatch.Close()
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
		}
		tmpPatch.Close()
		cmd := exec.CommandContext(ctx, "git", "apply", "--whitespace=fix", tmpPatch.Name())
		cmd.Dir = repoRoot
		out, err := cmd.CombinedOutput()
		if err != nil {
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: fmt.Sprintf("git apply failed: %v: %s", err, string(out))}
		}
		fmt.Printf("[diff] applied patch in %s\n", repoRoot)
		return OperationResult{Success: true, Op: "file", Path: m.Path}
	default:
		return OperationResult{Success: false, Op: "file", Path: m.Path, Message: "unsupported content_encoding"}
	}

	// verify checksum
	if m.Sha256 != "" {
		h := sha256.Sum256(data)
		if hex.EncodeToString(h[:]) != strings.ToLower(m.Sha256) {
			return OperationResult{Success: false, Op: "file", Path: m.Path, Message: "sha256 mismatch"}
		}
	}

	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
	}
	if err := os.WriteFile(abs, data, 0o644); err != nil {
		return OperationResult{Success: false, Op: "file", Path: m.Path, Message: err.Error()}
	}
	fmt.Printf("[diff] wrote %s\n", abs)
	return OperationResult{Success: true, Op: "file", Path: m.Path}
}

func (d *DefaultOperationHandler) HandleChunk(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	// Append chunk into a staging file under repoRoot/.stream/<fileID>
	if m.DataEncoding != "base64" {
		return OperationResult{Success: false, Op: "chunk", Message: "unsupported encoding", Extra: map[string]interface{}{"file_id": m.FileID}}
	}
	raw, err := base64.StdEncoding.DecodeString(m.Data)
	if err != nil {
		return OperationResult{Success: false, Op: "chunk", Message: err.Error(), Extra: map[string]interface{}{"file_id": m.FileID}}
	}
	outPath := filepath.Join(repoRoot, ".stream", m.FileID)
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return OperationResult{Success: false, Op: "chunk", Message: err.Error(), Extra: map[string]interface{}{"file_id": m.FileID}}
	}
	// append to file
	f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return OperationResult{Success: false, Op: "chunk", Message: err.Error(), Extra: map[string]interface{}{"file_id": m.FileID}}
	}
	_, err = f.Write(raw)
	f.Close()
	if err != nil {
		return OperationResult{Success: false, Op: "chunk", Message: err.Error(), Extra: map[string]interface{}{"file_id": m.FileID}}
	}
	fmt.Printf("[diff] appended chunk %d/%d for %s -> %s\n", m.ChunkIndex, m.TotalChunks, m.FileID, outPath)
	return OperationResult{Success: true, Op: "chunk", Extra: map[string]interface{}{"file_id": m.FileID}}
}

func (d *DefaultOperationHandler) HandleCommit(ctx context.Context, repoRoot string, m *StreamMessage) OperationResult {
	// On commit, if finalize, run git add/commit in repoRoot
	// Treat a bare commit (no finalize/message) as a finalize request so
	// providers that emit just {"type":"commit"} still cause a commit.
	finalize := m.Finalize
	if !finalize && m.Message == "" {
		finalize = true
	}
	if finalize {
		msg := m.Message
		if msg == "" {
			msg = "stream-commit"
		}
		cmd := exec.CommandContext(ctx, "git", "add", "-A")
		cmd.Dir = repoRoot
		if out, err := cmd.CombinedOutput(); err != nil {
			return OperationResult{Success: false, Op: "commit", Message: fmt.Sprintf("git add failed: %v: %s", err, string(out))}
		}
		cmd = exec.CommandContext(ctx, "git", "commit", "-m", m.Message)
		cmd.Dir = repoRoot
		if out, err := cmd.CombinedOutput(); err != nil {
			// ignore nothing to commit
			if bytes.Contains(out, []byte("nothing to commit")) {
				return OperationResult{Success: true, Op: "commit", Message: "nothing to commit"}
			}
			return OperationResult{Success: false, Op: "commit", Message: fmt.Sprintf("git commit failed: %v: %s", err, string(out))}
		}
		fmt.Printf("[diff] committed in %s: %s\n", repoRoot, m.Message)
		return OperationResult{Success: true, Op: "commit", Message: m.Message}
	}
	return OperationResult{Success: true, Op: "commit", Message: "noop"}
}

// SetHandler attaches an OperationHandler to the parser.
func (p *DiffParser) SetHandler(h OperationHandler) {
	p.handler = h
}

// PopReports returns and clears any reports produced by the handler.
func (p *DiffParser) PopReports() []OperationResult {
	p.mu.Lock()
	defer p.mu.Unlock()
	r := p.reports
	p.reports = nil
	return r
}

func (p *DiffParser) appendReport(r OperationResult) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reports = append(p.reports, r)
}

// SetFileHandler registers a custom handler for "file" messages.
func (p *DiffParser) SetFileHandler(h FileHandler) {
	p.fileHandler = h
}

// SetChunkHandler registers a custom handler for "chunk" messages.
func (p *DiffParser) SetChunkHandler(h StreamChunkHandler) {
	p.chunkHandler = h
}

// SetCommitHandler registers a custom handler for "commit" messages.
func (p *DiffParser) SetCommitHandler(h CommitHandler) {
	p.commitHandler = h
}

// ProcessChunk parses NDJSON lines found inside the provided accumulated response chunk
// and applies file operations synchronously. It expects chunk-level material: res.Chunk.Message.Content.
func (p *DiffParser) ProcessChunk(ctx context.Context, res *AccumulatedResponse) error {
	if res == nil || res.Chunk == nil {
		return nil
	}
	chunkText := res.Chunk.Message.Content
	if strings.TrimSpace(chunkText) == "" {
		return nil
	}

	// Each logical event is a JSON object on its own line (NDJSON).
	lines := strings.Split(chunkText, "\n")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		var msg StreamMessage
		if err := json.Unmarshal([]byte(l), &msg); err != nil {
			// Not a structured stream event; ignore and continue
			continue
		}

		if err := p.HandleMessage(ctx, &msg); err != nil {
			return err
		}
	}
	return nil
}

// HandleMessage processes a single parsed StreamMessage and dispatches to the
// configured handlers or built-in defaults. This allows external parsers (like
// FenceParser) to feed messages directly into the DiffParser.
func (p *DiffParser) HandleMessage(ctx context.Context, msg *StreamMessage) error {
	switch msg.Type {
	case "file":
		if p.handler != nil {
			rep := p.handler.HandleFile(ctx, p.repoRoot, msg)
			p.appendReport(rep)
			return nil
		}
		if p.fileHandler != nil {
			return p.fileHandler(ctx, msg)
		}
		return p.handleFile(ctx, msg)
	case "chunk":
		if p.handler != nil {
			rep := p.handler.HandleChunk(ctx, p.repoRoot, msg)
			p.appendReport(rep)
			return nil
		}
		if p.chunkHandler != nil {
			return p.chunkHandler(ctx, msg)
		}
		return p.handleChunk(ctx, msg)
	case "commit":
		if p.handler != nil {
			rep := p.handler.HandleCommit(ctx, p.repoRoot, msg)
			p.appendReport(rep)
			return nil
		}
		if p.commitHandler != nil {
			return p.commitHandler(ctx, msg)
		}
		// legacy behaviour: only run git commit when finalized and message provided
		if msg.Finalize && msg.Message != "" {
			_ = p.gitCommit(ctx, msg.Message)
		}
		return nil
	case "done", "abort", "meta", "init":
		// ignore these at the parser level
		return nil
	default:
		// unknown type -> ignore
		return nil
	}
}

// safe path resolution
func (p *DiffParser) absPathFor(rel string) (string, error) {
	// use slash-cleaning to allow NDJSON to send POSIX paths
	rel = filepath.ToSlash(rel)
	clean := filepath.Clean(rel)
	if filepath.IsAbs(clean) {
		return "", fmt.Errorf("absolute paths not allowed: %q", rel)
	}
	// prevent traversal beyond repoRoot
	if strings.HasPrefix(clean, ".."+string(filepath.Separator)) || clean == ".." {
		return "", fmt.Errorf("path traversal detected: %q", rel)
	}
	return filepath.Join(p.repoRoot, clean), nil
}

// default file handler (writes to disk or applies unified diff)
func (p *DiffParser) handleFile(ctx context.Context, m *StreamMessage) error {
	if m.Op == "delete" {
		target, err := p.absPathFor(m.Path)
		if err != nil {
			return err
		}
		_ = os.Remove(target)
		return nil
	}

	// Build bytes for content
	var data []byte
	switch m.ContentEncoding {
	case "base64":
		d, err := base64.StdEncoding.DecodeString(m.Content)
		if err != nil {
			return err
		}
		data = d
	case "utf-8", "":
		data = []byte(m.Content)
	case "unified":
		// apply unified diff (git apply)
		tmpPatch, err := os.CreateTemp("", "stream-*.patch")
		if err != nil {
			return err
		}
		defer os.Remove(tmpPatch.Name())
		if _, err := tmpPatch.WriteString(m.Content); err != nil {
			tmpPatch.Close()
			return err
		}
		tmpPatch.Close()
		cmd := exec.CommandContext(ctx, "git", "apply", "--whitespace=fix", tmpPatch.Name())
		cmd.Dir = p.repoRoot
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("git apply failed: %v: %s", err, string(out))
		}
		return nil
	default:
		return fmt.Errorf("unsupported content_encoding: %s", m.ContentEncoding)
	}

	// verify checksum if present
	if m.Sha256 != "" {
		h := sha256.Sum256(data)
		if hex.EncodeToString(h[:]) != strings.ToLower(m.Sha256) {
			return fmt.Errorf("sha256 mismatch for %s", m.Path)
		}
	}

	target, err := p.absPathFor(m.Path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}

	// atomic write if requested
	if m.Atomic {
		tmp, err := os.CreateTemp(filepath.Dir(target), ".tmp-*")
		if err != nil {
			return err
		}
		if _, err := tmp.Write(data); err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
			return err
		}
		tmp.Close()
		if err := os.Rename(tmp.Name(), target); err != nil {
			return err
		}
	} else {
		if err := os.WriteFile(target, data, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func (p *DiffParser) handleChunk(ctx context.Context, m *StreamMessage) error {
	if m.DataEncoding != "base64" {
		return fmt.Errorf("unsupported chunk encoding: %s", m.DataEncoding)
	}
	raw, err := base64.StdEncoding.DecodeString(m.Data)
	if err != nil {
		return err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.chunks[m.FileID]; !ok {
		p.chunks[m.FileID] = make(map[int][]byte)
	}
	p.chunks[m.FileID][m.ChunkIndex] = raw
	if m.TotalChunks > 0 {
		p.totals[m.FileID] = m.TotalChunks
	}

	// if we already know total and have all, assemble
	total, haveTotal := p.totals[m.FileID]
	if haveTotal {
		if len(p.chunks[m.FileID]) >= total {
			// assemble
			parts := make([][]byte, total)
			for i := 0; i < total; i++ {
				part, ok := p.chunks[m.FileID][i]
				if !ok {
					return fmt.Errorf("missing chunk %d for %s", i, m.FileID)
				}
				parts[i] = part
			}
			assembled := bytes.Join(parts, nil)
			// write assembled to staging path (.stream/<fileID>) so handlers can pick it up
			outPath := filepath.Join(p.repoRoot, ".stream", m.FileID)
			if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
				return err
			}
			if err := os.WriteFile(outPath, assembled, 0o644); err != nil {
				return err
			}
			// cleanup memory
			delete(p.chunks, m.FileID)
			delete(p.totals, m.FileID)
		}
	}
	return nil
}

func (p *DiffParser) gitCommit(ctx context.Context, message string) error {
	cmd := exec.CommandContext(ctx, "git", "add", "-A")
	cmd.Dir = p.repoRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git add failed: %v: %s", err, string(out))
	}
	cmd = exec.CommandContext(ctx, "git", "commit", "-m", message)
	cmd.Dir = p.repoRoot
	if out, err := cmd.CombinedOutput(); err != nil {
		// If there is nothing to commit, ignore error
		if bytes.Contains(out, []byte("nothing to commit")) {
			return nil
		}
		return fmt.Errorf("git commit failed: %v: %s", err, string(out))
	}
	return nil
}

// AttachParserToAccumulator composes a parser into an accumulated response stream.
// It returns a new channel that yields the same AccumulatedResponse values after
// the parser has processed each chunk. Errors are logged and don't close the stream.
func AttachParserToAccumulator(ctx context.Context, input <-chan *AccumulatedResponse, parser *DiffParser) <-chan *AccumulatedResponse {
	out := make(chan *AccumulatedResponse)
	go func() {
		defer close(out)
		for res := range input {
			if res != nil && parser != nil {
				if err := parser.ProcessChunk(ctx, res); err != nil {
					fmt.Printf("diff parser error: %v\n", err)
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
