To modify files, emit fenced `diff` blocks containing git unified diffs. Each block must identify the target file using standard `--- a/` and `+++ b/` headers.

For modifications, ensure your diff is well-formed so that standard `git apply` succeeds:
1. Use `--- a/path/to/file` and `+++ b/path/to/file` headers.
2. Every line in the hunk must start with ` `, `+`, or `-`.
3. Context lines (those starting with a space) must match the file content EXACTLY, including whitespace and indentation.
4. Provide at least 3 lines of context around changes when possible.
5. Include accurate hunk headers `@@ -start,len +start,len @@`.

Example:
```diff
--- a/path/to/file
+++ b/path/to/file
@@ -10,4 +10,5 @@
 context line
-removed line
+added line
 more context
```

For new files, use `--- /dev/null` and `+++ b/path`.

ALWAYS READ THE FILE CONTENT FIRST TO ENSURE CONTEXT MATCHES. You will receive a [diff-report] after processing.
