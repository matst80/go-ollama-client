package ai

import (
	"context"
	"fmt"
)

type AgentSessionOption func(*AgentSession)

type AgentSessionInterface interface {
	SendUserMessage(ctx context.Context, msg string) error
	Recv() <-chan AccumulatedResponse
	GetMessageHistory() []Message
	Stop()
}

// AgentSession manages a continuous chat session, tracking the underlying ChatRequest
// and providing a single global channel for responses of type T.
type AgentSession struct {
	client     ChatClientInterface
	rec        *ChatRequest
	globalChan chan AccumulatedResponse
	ctx        context.Context
	cancel     context.CancelFunc
}

// NewAgentSession creates a new AgentSession with the given client and base request.
// Use options like WithTransformer or WithAccumulator to specify the output type.
func NewAgentSession(ctx context.Context, client ChatClientInterface, req *ChatRequest, opts ...AgentSessionOption) *AgentSession {
	ctx, cancel := context.WithCancel(ctx)
	session := &AgentSession{
		client: client,
		rec:    req,
		// We can't pre-allocate GlobalChan with T if we don't know the capacity needed,
		// but 100 is a safe default for streaming.
		globalChan: make(chan AccumulatedResponse, 100),
		ctx:        ctx,
		cancel:     cancel,
	}
	for _, opt := range opts {
		opt(session)
	}
	return session
}

// Recv returns a receive-only channel for all chat responses across multiple turns.
func (a *AgentSession) Recv() <-chan AccumulatedResponse {
	return a.globalChan
}

// Stop cancels the session context and closes the global channel.
func (a *AgentSession) Stop() {
	a.cancel()
	close(a.globalChan)
}

// SendUserMessage appends a user message to the request and triggers a streaming chat.
func (a *AgentSession) SendUserMessage(ctx context.Context, msg string) error {
	a.rec.AddMessage(MessageRoleUser, msg)
	return a.streamChat(ctx)
}

// SendMessages appends one or more pre-constructed messages (like tool results) and triggers a streaming chat.
func (a *AgentSession) SendMessages(ctx context.Context, msgs ...Message) error {
	a.rec.Messages = append(a.rec.Messages, msgs...)
	return a.streamChat(ctx)
}

// GetMessageHistory returns the current conversation history.
func (a *AgentSession) GetMessageHistory() []Message {
	return a.rec.Messages
}

// streamChat performs the streamed chat request and pipes the results into GlobalChan.
func (a *AgentSession) streamChat(ctx context.Context) error {
	ch := make(chan *ChatResponse)

	// Start the request in a goroutine
	go func() {
		if err := a.client.ChatStreamed(ctx, *a.rec, ch); err != nil {
			fmt.Printf("ChatStreamed error: %v\n", err)
		}
	}()

	// Use StreamAccumulator to get accumulated responses
	accCh := StreamAccumulator(ctx, ch, false)

	go func() {
		var last *AccumulatedResponse
		for res := range accCh {
			last = res
			select {
			case a.globalChan <- *res:
			case <-a.ctx.Done():
				return
			case <-ctx.Done():
				return
			}
		}

		// When the stream is finished, append the assistant's final accumulated response to history
		if last != nil {
			a.rec.Messages = append(a.rec.Messages, Message{
				Role:             MessageRoleAssistant,
				Content:          last.Content,
				ReasoningContent: last.ReasoningContent,
				ToolCalls:        last.ToolCalls,
			})
		}
	}()

	return nil
}
