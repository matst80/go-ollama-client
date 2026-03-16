Fenced diffstream support
=========================

This repository now accepts machine-actionable streamed file edits using a simple, model-friendly fenced format called `diffstream` (in addition to the prior NDJSON path). The fenced format avoids JSON escaping and is easier for models to emit reliably.

Key points
- Preferred input: fenced blocks that start with ```diffstream and include header attributes on the opening line (e.g. `type=file op=add path=workspace/info.txt encoding=utf-8`).
- Supported operations: `file` (add/update/delete/patch), `chunk` (multi-part uploads), `commit` (finalize), and meta/control messages like `done`/`init`/`meta`.
- Chunk fences can use `data_encoding=utf-8` for chunked text (recommended in examples) or `data_encoding=base64` for binary uploads — base64 is supported but omitted from examples to keep them simple.
- AgentSession is wired to use the FenceParser by default; the streamed parser will apply operations to a `repoRoot` and report results as `[diff-report]` when the stream ends.

Fenced format overview
- Opening fence: ```diffstream [headers...]
- Body: any text representing file contents, chunk data, or a unified patch
- Closing fence: ```

Header attributes (common)
- `type` — one of `file`, `chunk`, `commit`, `meta`.
- `op` — for `type=file`: `add`, `update`, `delete`, `patch`.
- `path` — relative POSIX path for file operations.
- `encoding` or `content_encoding` — `utf-8` or `unified` (for unified diffs).
- `file_id`, `chunk_index`, `total_chunks`, `data_encoding` — for chunked uploads.
- `message`, `finalize` — for commits.

Examples

Add a small UTF-8 text file (recommended):

```diffstream type=file op=add path=workspace/info.txt encoding=utf-8
The capital of France is Paris.
```

Chunked text upload (split into two fenced chunks):

```diffstream type=chunk file_id=f1 chunk_index=0 total_chunks=2 data_encoding=utf-8
First half of a large text file...
```
```diffstream type=chunk file_id=f1 chunk_index=1 total_chunks=2 data_encoding=utf-8
Second half of the large text file...
```

Commit after edits (finalize):

```diffstream type=commit message="Add info.txt" finalize=true

```

Notes on NDJSON and base64
- The system still supports NDJSON lines (each line is a JSON object) for backwards compatibility, but the README examples now show the fenced format only. Models may still emit base64 chunk data when necessary; the parser accepts `data_encoding=base64`.

Agent integration (how to enable)
- The session constructor supports configuration options:

  - `ai.WithRepoRoot(path)` — set the repository/workspace root where file operations are applied.
  - `ai.WithOperationHandler(handler)` — provide a custom `OperationHandler` implementation. The repo already includes `ai.DefaultOperationHandler` that writes files, reassembles chunks under `.stream/`, runs `git apply` for unified diffs and `git add`/`commit` on commit requests.

Example: wire the master session to use a repo and the default handler (see `main.go`):

```go
masterSession := ai.NewAgentSession(ctx, masterClient, masterReq,
    ai.WithRepoRoot("./test-repo"),
    ai.WithOperationHandler(&ai.DefaultOperationHandler{}),
)
```

Tests
- Unit tests for the fence parser live in `pkg/ai/fence_parser_test.go`.
- An integration test demonstrating fenced file write + commit is `pkg/ai/fence_integration_test.go`.
- Run tests with:

```bash
go test ./pkg/ai
```

Files of interest
- `pkg/ai/fence_parser.go` — fence parsing implementation.
- `pkg/ai/diffstream.go` — DiffParser, message handling, default operation handler.
- `main.go` — example master session and prompts illustrating usage.

Next steps / suggestions
- If you want NDJSON removed entirely from code paths, we can deprecate that parser and provide a configuration option to enable NDJSON-only sessions.
- Add README examples for advanced scenarios (binary chunking with base64, unified diffs) if needed for clients.

If you want me to add the README content to another file or expand any example, tell me which section to expand.
