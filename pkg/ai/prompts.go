package ai

// SystemPromptGitDiff is the preferred reusable system prompt. It instructs models to
// emit fenced `diff` blocks containing exact git unified diffs.
var SystemPromptGitDiff = "To modify files, use fenced `diff` blocks. Each block must identify the target file using standard `---` and `+++` headers.\n" +
	"For modifications, use this form:\n" +
	"```diff\n" +
	"--- a/path/to/file\n" +
	"+++ b/path/to/file\n" +
	"@@ ...\n" +
	"...diff content...\n" +
	"```\n" +
	"The system supports standard unified git diffs and can fuzzily apply \"lazy\" diffs (even those missing `@@` headers) as long as the file path is correct.\n" +
	"For new files, use `/dev/null` for the source side, for example:\n" +
	"```diff\n" +
	"--- /dev/null\n" +
	"+++ b/new-file.txt\n" +
	"@@ -0,0 +1,1 @@\n" +
	"+content for the new file\n" +
	"```\n" +
	"You are encouraged to provide explanations and answer questions normally." +
	"ALWAYS READ THE FILE CONTENT FIRST TO ENSURE CONTEXT MATCHES PERFECTLY. " +
	"After processing, the system will emit a [diff-report] summary of the operations."
