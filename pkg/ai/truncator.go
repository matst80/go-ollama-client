package ai

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// TruncationStrategy defines the interface for message history truncation strategies.
type TruncationStrategy interface {
	// Apply takes the current messages and returns truncated messages along with count of removed messages.
	Apply(messages []Message) ([]Message, int)
}

// MiddleTruncator removes a fixed number of middle messages when the total exceeds a threshold.
// It always preserves system/user prompts at the beginning and never removes the last message.
type MiddleTruncator struct {
	Threshold   int // Minimum message count before truncation kicks in (> Threshold triggers truncation)
	RemoveCount int // Number of messages to remove from the middle
}

// Apply implements TruncationStrategy for MiddleTruncator.
// It keeps first N and last M messages while removing ~RemoveCount from the middle.
func (m *MiddleTruncator) Apply(messages []Message) ([]Message, int) {
	if len(messages) <= m.Threshold {
		return messages, 0
	}
	// Calculate how many non-essential messages we can remove
	totalToRemove := m.RemoveCount

	// Always preserve the last message (never remove it)
	lastMessageIndex := len(messages) - 1

	// Identify removable messages (not system/user and not the last message)
	removableIndices := make([]int, 0)
	for i := range messages {
		if i != lastMessageIndex && !isSystemOrUserMessage(messages[i]) {
			removableIndices = append(removableIndices, i)
		}
	}

	// If we don't have enough removable messages, skip truncation and log warning
	if len(removableIndices) < totalToRemove {
		fmt.Printf("[Truncator] Warning: Not enough removable messages to remove %d (only %d available). Skipping truncation.\n",
			totalToRemove, len(removableIndices))
		return messages, 0
	}

	// Select middle messages for removal (avoiding system/user positions)
	toRemove := m.selectMiddleMessages(messages, totalToRemove, removableIndices, lastMessageIndex)

	// Build result by keeping non-removed messages
	result := make([]Message, 0, len(messages)-len(toRemove))
	removedCount := 0

	for i, msg := range messages {
		if !m.isInSlice(i, toRemove) {
			result = append(result, msg)
		} else {
			removedCount++
			fmt.Printf("[Truncator] Removed message %d (role: %s): %q\n", i, msg.Role, truncateContent(msg.Content, 50))
		}
	}

	return result, removedCount
}

// selectMiddleMessages selects which messages to remove from the middle.
func (m *MiddleTruncator) selectMiddleMessages(messages []Message, count int, removableIndices []int, lastMsgIdx int) []int {
	if len(removableIndices) == 0 {
		return nil
	}

	// Sort indices to work with them in order
	selected := make([]int, 0, count)

	// Select messages from the middle region
	for i, idx := range removableIndices {
		if idx == lastMsgIdx {
			continue
		}

		// Prefer removing from center of conversation (not too early, not too late)
		isEarly := i < len(removableIndices)/3
		isLate := i > 2*len(removableIndices)/3

		if !isEarly && !isLate {
			selected = append(selected, idx)
			if len(selected) >= count {
				break
			}
		}
	}

	// If we haven't selected enough, fill from remaining positions
	if len(selected) < count {
		for _, idx := range removableIndices {
			if idx == lastMsgIdx {
				continue
			}
			if !m.isInSlice(idx, selected) {
				selected = append(selected, idx)
				if len(selected) >= count {
					break
				}
			}
		}
	}

	return selected[:count]
}

// isInSlice checks if an integer is in a slice.
func (m *MiddleTruncator) isInSlice(target int, slice []int) bool {
	for _, v := range slice {
		if v == target {
			return true
		}
	}
	return false
}

// isSystemOrUserMessage checks if a message has system or user role.
func isSystemOrUserMessage(msg Message) bool {
	return msg.Role == MessageRoleSystem || msg.Role == MessageRoleUser
}

// truncateContent truncates long content for logging purposes.
func truncateContent(content string, maxLen int) string {
	if len(content) <= maxLen {
		return content
	}
	return content[:maxLen] + "..."
}

// SummarizeTruncator summarizes a block of messages by sending them to the
// configured ChatClientInterface and replacing the block with a single summary
// message. It follows the same high-level pattern as MiddleTruncator but calls
// out to a client to produce the summary.
type SummarizeTruncator struct {
	Threshold     int                 // Minimum message count before summarization kicks in
	RemoveCount   int                 // Number of messages to summarize (approx)
	Client        ChatClientInterface // Client used to obtain the summary
	Model         string              // model to use for summarization (optional)
	SummaryPrompt string              // system prompt to guide summarization
	Timeout       time.Duration       // timeout for the summarization call
	// TokenEstimateThreshold triggers a pre-compression step when the
	// estimated token count of the selected messages exceeds this value.
	// If zero, no pre-compression is performed.
	TokenEstimateThreshold int
	Logger                 Logger
}

// SummarizeOptions configures a SummarizeTruncator when created via constructor.
type SummarizeOptions struct {
	Threshold              int
	RemoveCount            int
	Model                  string
	SummaryPrompt          string
	Timeout                time.Duration
	TokenEstimateThreshold int
	Logger                 Logger
}

// NewSummarizeTruncator creates a SummarizeTruncator with sensible defaults.
// If opts is nil, default values are applied. Client may be nil to disable
// summarization (Apply will be a no-op).
func NewSummarizeTruncator(client ChatClientInterface, opts *SummarizeOptions) *SummarizeTruncator {
	var o SummarizeOptions
	if opts != nil {
		o = *opts
	}
	if o.Threshold == 0 {
		o.Threshold = 50
	}
	if o.RemoveCount == 0 {
		o.RemoveCount = 10
	}
	if o.Timeout == 0 {
		o.Timeout = 10 * time.Second
	}
	if o.TokenEstimateThreshold == 0 {
		o.TokenEstimateThreshold = 2000
	}
	return &SummarizeTruncator{
		Threshold:              o.Threshold,
		RemoveCount:            o.RemoveCount,
		Client:                 client,
		Model:                  o.Model,
		SummaryPrompt:          o.SummaryPrompt,
		Timeout:                o.Timeout,
		TokenEstimateThreshold: o.TokenEstimateThreshold,
		Logger:                 o.Logger,
	}
}

// Apply implements TruncationStrategy for SummarizeTruncator.
func (s *SummarizeTruncator) Apply(messages []Message) ([]Message, int) {
	if s.Client == nil {
		// No client configured; behave like MiddleTruncator by doing nothing
		if s.Logger != nil {
			s.Logger.Warnf("[SummarizeTruncator] No client configured. Skipping summarization.")
		}
		return messages, 0
	}

	if len(messages) <= s.Threshold {
		return messages, 0
	}

	totalToRemove := s.RemoveCount
	lastMessageIndex := len(messages) - 1

	// Identify removable indices (not system/user and not the last message)
	removableIndices := make([]int, 0)
	for i := range messages {
		if i != lastMessageIndex && !isSystemOrUserMessage(messages[i]) {
			removableIndices = append(removableIndices, i)
		}
	}

	if len(removableIndices) == 0 {
		if s.Logger != nil {
			s.Logger.Infof("[SummarizeTruncator] No removable messages available.")
		}
		return messages, 0
	}

	// Select indices to summarize (prefer middle)
	toRemove := s.selectMiddleMessages(messages, totalToRemove, removableIndices, lastMessageIndex)

	// Build chat request: include a guiding system prompt and the messages to summarize
	req := NewChatRequest(s.Model)
	if s.SummaryPrompt != "" {
		req.AddMessage(MessageRoleSystem, s.SummaryPrompt)
	} else {
		req.AddMessage(MessageRoleSystem, "Summarize the following conversation into a concise summary with key points and actions.")
	}

	// Append the messages to be summarized in chronological order
	// Use AddMessageStruct which copies the Message struct
	// We only want the content and role as context for summarization
	// Append in order of indices
	sorted := make([]int, len(toRemove))
	copy(sorted, toRemove)
	// simple sort (they are already in ascending order from selectMiddleMessages, but ensure)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	for _, idx := range sorted {
		req.AddMessageStruct(&messages[idx])
	}

	// If token estimation threshold is configured, estimate tokens for the block
	// and perform a pre-compression step if it exceeds the threshold.
	if s.TokenEstimateThreshold > 0 {
		estimated := estimateTokenCountForIndices(messages, sorted)
		if estimated > s.TokenEstimateThreshold {
			if s.Logger != nil {
				s.Logger.Infof("[SummarizeTruncator] Estimated tokens %d > threshold %d: performing pre-compression", estimated, s.TokenEstimateThreshold)
			} else {
				fmt.Printf("[SummarizeTruncator] Estimated tokens %d > threshold %d: performing pre-compression\n", estimated, s.TokenEstimateThreshold)
			}

			compReq := NewChatRequest(s.Model)
			compReq.AddMessage(MessageRoleSystem, "Compress the following conversation into a short, high-density summary (aim for <= 200 tokens). Keep only key facts and actions.")
			for _, idx := range sorted {
				compReq.AddMessageStruct(&messages[idx])
			}

			compCtx, compCancel := context.WithTimeout(context.Background(), s.Timeout)
			compResp, compErr := s.Client.Chat(compCtx, *compReq)
			compCancel()
			if compErr == nil && compResp != nil && compResp.Message.Content != "" {
				// Replace the detailed block with a single user message containing the compressed text
				compressedMsg := Message{Role: MessageRoleUser, Content: compResp.Message.Content}
				// Rebuild req to contain only system prompt + compressed content
				req = NewChatRequest(s.Model)
				if s.SummaryPrompt != "" {
					req.AddMessage(MessageRoleSystem, s.SummaryPrompt)
				} else {
					req.AddMessage(MessageRoleSystem, "Summarize the following conversation into a concise summary with key points and actions.")
				}
				req.AddMessageStruct(&compressedMsg)
			} else {
				if s.Logger != nil {
					s.Logger.Warnf("[SummarizeTruncator] Pre-compression failed: %v; continuing with original block", compErr)
				} else {
					fmt.Printf("[SummarizeTruncator] Pre-compression failed: %v; continuing with original block\n", compErr)
				}
			}
		}
	}

	// Call the client with a timeout
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	resp, err := s.Client.Chat(ctx, *req)
	if err != nil || resp == nil || resp.Message.Content == "" {
		if s.Logger != nil {
			s.Logger.Warnf("[SummarizeTruncator] Summarization failed: %v. Falling back to truncation.", err)
		} else {
			fmt.Printf("[SummarizeTruncator] Summarization failed: %v. Falling back to truncation.\n", err)
		}
		// Fallback: simply remove the selected messages (like MiddleTruncator)
		result := make([]Message, 0, len(messages)-len(toRemove))
		removedCount := 0
		toRemoveMap := make(map[int]struct{}, len(toRemove))
		for _, v := range toRemove {
			toRemoveMap[v] = struct{}{}
		}
		for i, msg := range messages {
			if _, ok := toRemoveMap[i]; !ok {
				result = append(result, msg)
			} else {
				removedCount++
				if s.Logger != nil {
					s.Logger.Infof("[SummarizeTruncator] Removed message %d (role: %s): %q", i, msg.Role, truncateContent(msg.Content, 50))
				} else {
					fmt.Printf("[SummarizeTruncator] Removed message %d (role: %s): %q\n", i, msg.Role, truncateContent(msg.Content, 50))
				}
			}
		}
		return result, removedCount
	}

	summaryContent := fmt.Sprintf("[Summary of %d messages]\n\n%s", len(toRemove), resp.Message.Content)
	summaryMsg := Message{Role: MessageRoleAssistant, Content: summaryContent}

	// New behavior: after successful summarization, preserve only system prompts
	// and the generated summary message.
	result := make([]Message, 0, 1+len(messages))
	for _, msg := range messages {
		if msg.Role == MessageRoleSystem {
			result = append(result, msg)
		}
	}
	result = append(result, summaryMsg)

	removedCount := len(messages) - len(result)
	if s.Logger != nil {
		s.Logger.Infof("[SummarizeTruncator] Summarized %d messages into 1 summary; preserved %d system messages.", len(toRemove), len(result)-1)
	} else {
		fmt.Printf("[SummarizeTruncator] Summarized %d messages into 1 summary; preserved %d system messages.\n", len(toRemove), len(result)-1)
	}
	return result, removedCount
}

// estimateTokenCountForIndices provides a very rough token estimate for the
// concatenation of message contents at the given indices. This is intentionally
// conservative and cheap: we estimate tokens by counting words and applying a
// multiplier to approximate subword tokenization. It's not exact but is fine
// for deciding whether a pre-compression step should run.
func estimateTokenCountForIndices(messages []Message, indices []int) int {
	if len(indices) == 0 {
		return 0
	}
	totalWords := 0
	for _, idx := range indices {
		if idx < 0 || idx >= len(messages) {
			continue
		}
		// split on spaces
		totalWords += len(splitWords(messages[idx].Content))
	}
	// heuristic: average tokens per word ~= 1.3 (subword tokens)
	return int(float64(totalWords) * 1.3)
}

func splitWords(s string) []string {
	if s == "" {
		return nil
	}
	// simple split by whitespace
	fields := make([]string, 0)
	word := ""
	for _, r := range s {
		if r == ' ' || r == '\n' || r == '\t' || r == '\r' {
			if word != "" {
				fields = append(fields, word)
				word = ""
			}
			continue
		}
		word += string(r)
	}
	if word != "" {
		fields = append(fields, word)
	}
	return fields
}

// selectMiddleMessages selects which messages to remove from the middle.
func (s *SummarizeTruncator) selectMiddleMessages(messages []Message, count int, removableIndices []int, lastMsgIdx int) []int {
	if len(removableIndices) == 0 {
		return nil
	}

	selected := make([]int, 0, count)

	for i, idx := range removableIndices {
		if idx == lastMsgIdx {
			continue
		}

		isEarly := i < len(removableIndices)/3
		isLate := i > 2*len(removableIndices)/3

		if !isEarly && !isLate {
			selected = append(selected, idx)
			if len(selected) >= count {
				break
			}
		}
	}

	if len(selected) < count {
		for _, idx := range removableIndices {
			if idx == lastMsgIdx {
				continue
			}
			if !s.isInSlice(idx, selected) {
				selected = append(selected, idx)
				if len(selected) >= count {
					break
				}
			}
		}
	}

	if len(selected) > count {
		return selected[:count]
	}
	return selected
}

// isInSlice checks if an integer is in a slice.
func (s *SummarizeTruncator) isInSlice(target int, slice []int) bool {
	for _, v := range slice {
		if v == target {
			return true
		}
	}
	return false
}

// AgeTruncator removes messages older than MaxAge.
// It preserves system messages and the last message.
type AgeTruncator struct {
	MaxAge    time.Duration
	Threshold int // Optional: Only truncate if total messages > Threshold
	Logger    Logger
}

func NewAgeTruncator(maxAge time.Duration, threshold int, logger Logger) *AgeTruncator {
	return &AgeTruncator{
		MaxAge:    maxAge,
		Threshold: threshold,
		Logger:    logger,
	}
}

func (a *AgeTruncator) Apply(messages []Message) ([]Message, int) {
	if len(messages) <= a.Threshold {
		return messages, 0
	}

	now := time.Now()
	lastIdx := len(messages) - 1
	result := make([]Message, 0, len(messages))
	removed := 0

	for i, msg := range messages {
		// Always keep system messages and the last message
		if msg.Role == MessageRoleSystem || i == lastIdx {
			result = append(result, msg)
			continue
		}

		// Check age
		if !msg.CreatedAt.IsZero() && now.Sub(msg.CreatedAt) > a.MaxAge {
			removed++
			if a.Logger != nil {
				a.Logger.Infof("[AgeTruncator] Removed expired message (age: %v): %s", now.Sub(msg.CreatedAt), truncateContent(msg.Content, 50))
			}
			continue
		}

		result = append(result, msg)
	}

	return result, removed
}

// MemoryStore defines the interface for long-term message storage.
type MemoryStore interface {
	AddMessages(messages []Message) error
	RetrieveRelevant(query string, limit int) ([]Message, error)
}

// InMemoryMemoryStore is a simple implementation for demonstration/testing.
type InMemoryMemoryStore struct {
	mu       sync.RWMutex
	messages []Message
}

func NewInMemoryMemoryStore() *InMemoryMemoryStore {
	return &InMemoryMemoryStore{}
}

func (s *InMemoryMemoryStore) AddMessages(messages []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messages = append(s.messages, messages...)
	return nil
}

func (s *InMemoryMemoryStore) RetrieveRelevant(query string, limit int) ([]Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Basic implementation: return the last N messages
	// In a real implementation this would use vector embeddings/search.
	if len(s.messages) <= limit {
		return s.messages, nil
	}
	return s.messages[len(s.messages)-limit:], nil
}

// MemoryTruncator wraps a strategy and stores removed messages in a MemoryStore.
type MemoryTruncator struct {
	Strategy TruncationStrategy
	Store    MemoryStore
	Logger   Logger
}

func NewMemoryTruncator(strategy TruncationStrategy, store MemoryStore, logger Logger) *MemoryTruncator {
	return &MemoryTruncator{
		Strategy: strategy,
		Store:    store,
		Logger:   logger,
	}
}

func (m *MemoryTruncator) Apply(messages []Message) ([]Message, int) {
	truncated, removedCount := m.Strategy.Apply(messages)
	if removedCount == 0 {
		return messages, 0
	}

	// Identify what was removed
	// Since Apply might return a modified slice (e.g. SummarizeTruncator returns a new summary),
	// we identify removed messages by comparing with the original set.
	removed := make([]Message, 0, removedCount)
	for _, orig := range messages {
		found := false
		for _, t := range truncated {
			// Basic heuristic match: Role, Content, and CreatedAt
			if orig.Role == t.Role && orig.Content == t.Content && orig.CreatedAt.Equal(t.CreatedAt) {
				found = true
				break
			}
		}
		if !found {
			removed = append(removed, orig)
		}
	}

	if len(removed) > 0 {
		if err := m.Store.AddMessages(removed); err != nil && m.Logger != nil {
			m.Logger.Warnf("[MemoryTruncator] Failed to store truncated messages: %v", err)
		} else if m.Logger != nil {
			m.Logger.Infof("[MemoryTruncator] Stored %d truncated messages in memory.", len(removed))
		}
	}

	return truncated, removedCount
}

// CompositeTruncator allows chaining multiple truncation strategies.
type CompositeTruncator struct {
	Strategies []TruncationStrategy
}

func NewCompositeTruncator(strategies ...TruncationStrategy) *CompositeTruncator {
	return &CompositeTruncator{Strategies: strategies}
}

func (c *CompositeTruncator) Apply(messages []Message) ([]Message, int) {
	totalRemoved := 0
	current := messages
	for _, s := range c.Strategies {
		next, removed := s.Apply(current)
		current = next
		totalRemoved += removed
	}
	return current, totalRemoved
}

// AutomaticMemoryHook is a session hook that automatically retrieves relevant
// memories and injects them as a system context message.
type AutomaticMemoryHook struct {
	DefaultSessionHooks
	Store       MemoryStore
	MaxMemories int
}

func NewAutomaticMemoryHook(store MemoryStore, maxMemories int) *AutomaticMemoryHook {
	return &AutomaticMemoryHook{
		Store:       store,
		MaxMemories: maxMemories,
	}
}

func (h *AutomaticMemoryHook) OnChatRequest(ctx context.Context, req *ChatRequest) error {
	if len(req.Messages) == 0 {
		return nil
	}

	// Use the last user message as a query
	var query string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == MessageRoleUser {
			query = req.Messages[i].Content
			break
		}
	}

	if query == "" {
		return nil
	}

	memories, err := h.Store.RetrieveRelevant(query, h.MaxMemories)
	if err != nil || len(memories) == 0 {
		return nil
	}

	// Format memories into a single context block
	var sb strings.Builder
	fmt.Fprintf(&sb, "Relevant context from previous conversations:\n")
	for _, m := range memories {
		fmt.Fprintf(&sb, "- [%v] %s\n", m.CreatedAt.Format("2006-01-02"), truncateContent(m.Content, 200))
	}

	// Inject as a system message at the beginning (after the main system prompt if it exists)
	contextMsg := Message{
		Role:      MessageRoleSystem,
		Content:   sb.String(),
		CreatedAt: time.Now(),
	}

	// Insert after first system message, or at the beginning
	insertIdx := 0
	if len(req.Messages) > 0 && req.Messages[0].Role == MessageRoleSystem {
		insertIdx = 1
	}

	newMsgs := make([]Message, 0, len(req.Messages)+1)
	newMsgs = append(newMsgs, req.Messages[:insertIdx]...)
	newMsgs = append(newMsgs, contextMsg)
	newMsgs = append(newMsgs, req.Messages[insertIdx:]...)
	req.Messages = newMsgs

	return nil
}

// ThinkingTruncator strips reasoning content (thinking) from all but the last assistant message.
type ThinkingTruncator struct{}

func (t *ThinkingTruncator) Apply(messages []Message) ([]Message, int) {
	lastAssistantWithThinkingIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == MessageRoleAssistant && messages[i].ReasoningContent != "" {
			lastAssistantWithThinkingIdx = i
			break
		}
	}

	if lastAssistantWithThinkingIdx == -1 {
		return messages, 0
	}

	modifiedCount := 0
	result := make([]Message, len(messages))
	for i, msg := range messages {
		if i != lastAssistantWithThinkingIdx && msg.ReasoningContent != "" {
			msg.ReasoningContent = ""
			modifiedCount++
			fmt.Printf("[ThinkingTruncator] Stripped thinking from message %d (role: %s)\n", i, msg.Role)
		}
		result[i] = msg
	}

	return result, 0
}

// RoundTruncator groups messages into rounds (starting with a User message)
// and preserves only the original User query and the final Assistant response
// for older rounds when the total message count exceeds a threshold.
type RoundTruncator struct {
	Threshold int // Minimum total messages before truncation kicks in
}

func (r *RoundTruncator) Apply(messages []Message) ([]Message, int) {
	if len(messages) <= r.Threshold {
		return messages, 0
	}

	result := make([]Message, 0, len(messages))

	// 1. Keep all system messages (usually at the beginning)
	for _, msg := range messages {
		if msg.Role == MessageRoleSystem {
			result = append(result, msg)
		}
	}

	// 2. Identify rounds (starting with a User message)
	type round struct {
		startIndex int
		endIndex   int // inclusive
	}
	rounds := make([]round, 0)
	var currentRound *round
	for i, msg := range messages {
		if msg.Role == MessageRoleSystem {
			continue
		}
		if msg.Role == MessageRoleUser {
			if currentRound != nil {
				currentRound.endIndex = i - 1
				rounds = append(rounds, *currentRound)
			}
			currentRound = &round{startIndex: i}
		}
	}
	if currentRound != nil {
		currentRound.endIndex = len(messages) - 1
		rounds = append(rounds, *currentRound)
	}

	// Always keep the last 2 rounds fully intact.
	keepRoundsCount := 2
	if len(rounds) <= keepRoundsCount {
		// Just append all non-system messages as they were
		for _, msg := range messages {
			if msg.Role != MessageRoleSystem {
				result = append(result, msg)
			}
		}
		return result, 0
	}

	// 3. Process rounds
	for ri, rnd := range rounds {
		// Latest rounds: keep entirely
		if ri >= len(rounds)-keepRoundsCount {
			for i := rnd.startIndex; i <= rnd.endIndex; i++ {
				result = append(result, messages[i])
			}
			continue
		}

		// Older rounds: keep User query + final Assistant response
		// Keep User query
		result = append(result, messages[rnd.startIndex])

		// Find and keep the last Assistant response specifically
		lastAssistantIdx := -1
		for i := rnd.endIndex; i > rnd.startIndex; i-- {
			if messages[i].Role == MessageRoleAssistant {
				lastAssistantIdx = i
				break
			}
		}

		if lastAssistantIdx != -1 {
			msg := messages[lastAssistantIdx]
			msg.ToolCalls = nil // Results are gone, so clear tool calls
			result = append(result, msg)
		}
	}

	removedCount := len(messages) - len(result)
	if removedCount > 0 {
		fmt.Printf("[RoundTruncator] Truncated history from %d to %d messages\n", len(messages), len(result))
	}

	return result, removedCount
}
