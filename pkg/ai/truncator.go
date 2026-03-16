package ai

import "fmt"

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
