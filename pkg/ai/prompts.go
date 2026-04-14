package ai

import (
	_ "embed"
	"strings"
)

//go:embed prompts/git_diff.md
var systemPromptGitDiff string

var SystemPromptGitDiff = strings.TrimSpace(systemPromptGitDiff)

