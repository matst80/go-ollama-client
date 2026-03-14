package ai

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type AgentSessionOption func(*AgentSession)

type AgentSessionInterface interface {
	SendUserMessage(ctx context.Context, msg string) error
	SendMessages(ctx context.Context, msgs ...Message) error
	Recv() <-chan AccumulatedResponse
	GetMessageHistory() []Message
	Stop()
	GetState() AgentState
	SetState(update func(*AgentState))
}

// AgentState holds the current status and metadata of an agent session
type AgentState struct {
	Status        string
	CurrentOutput string
	ParentID      string
	Title         string
	Type          string
	CreatedAt     time.Time
	LastActive    time.Time
}

// AgentSession manages a continuous chat session, tracking the underlying ChatRequest
// and providing a single global channel for responses of type T.
type AgentSession struct {
	client     ChatClientInterface
	rec        *ChatRequest
	globalChan chan AccumulatedResponse
	ctx        context.Context
	cancel     context.CancelFunc
	mu         sync.Mutex
	wg         sync.WaitGroup
	state      AgentState
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
		state: AgentState{
			Status:     "idle",
			CreatedAt:  time.Now(),
			LastActive: time.Now(),
		},
		wg: sync.WaitGroup{},
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
// It waits for any previous stream to finish.
func (a *AgentSession) SendUserMessage(ctx context.Context, msg string) error {
	a.wg.Wait()
	a.mu.Lock()
	defer a.mu.Unlock()
	a.rec.AddMessage(MessageRoleUser, msg)
	return a.streamChat(ctx)
}

// SendMessages appends one or more pre-constructed messages (like tool results) and triggers a streaming chat.
// It waits for any previous stream to finish.
func (a *AgentSession) SendMessages(ctx context.Context, msgs ...Message) error {
	a.wg.Wait()
	a.mu.Lock()
	defer a.mu.Unlock()
	a.rec.Messages = append(a.rec.Messages, msgs...)
	return a.streamChat(ctx)
}

// GetMessageHistory returns the current conversation history.
func (a *AgentSession) GetMessageHistory() []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.rec.Messages
}

func (a *AgentSession) GetState() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

func (a *AgentSession) SetState(update func(*AgentState)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	update(&a.state)
}

// streamChat performs the streamed chat request and pipes the results into GlobalChan.
func (a *AgentSession) streamChat(ctx context.Context) error {
	ch := make(chan *ChatResponse)

	// Start the request in a goroutine with retry logic
	go func() {
		defer close(ch)

		maxRetries := 3
		var lastErr error
		for i := 0; i < maxRetries; i++ {
			if i > 0 {
				fmt.Printf("Retrying ChatStreamed (%d/%d)... error was: %v\n", i, maxRetries, lastErr)
			}

			// Create a temporary channel for this attempt
			tempCh := make(chan *ChatResponse)

			// We need a separate goroutine because ChatStreamed is blocking
			go func() {
				a.SetState(func(as *AgentState) {
					as.Status = "waiting"
					as.LastActive = time.Now()
				})
				defer a.SetState(func(as *AgentState) {
					as.Status = "idle"
					as.LastActive = time.Now()
				})
				err := a.client.ChatStreamed(ctx, *a.rec, tempCh)
				if err != nil {
					lastErr = err
				}
			}()

			// Pipe tempCh to ch
			success := false
			for res := range tempCh {
				ch <- res
				success = true
			}

			if success && lastErr == nil {
				return
			}

			// If context is cancelled, don't retry
			if ctx.Err() != nil {
				break
			}
		}

		if lastErr != nil {
			errStr := lastErr.Error()
			fmt.Printf("ChatStreamed failed after %d retries: %v\n", maxRetries, lastErr)
			ch <- &ChatResponse{
				BaseResponse: &BaseResponse{
					Error: &errStr,
					Done:  true,
				},
			}
		}
	}()

	// Use StreamAccumulator to get accumulated responses
	accCh := StreamAccumulator(ctx, ch, false)

	a.wg.Add(1)
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
		// but only if it was successful (not an error and has content or tool calls)
		if last != nil && (last.Chunk == nil || last.Chunk.Error == nil) {
			a.mu.Lock()
			a.rec.Messages = append(a.rec.Messages, Message{
				Role:             MessageRoleAssistant,
				Content:          last.Content,
				ReasoningContent: last.ReasoningContent,
				ToolCalls:        last.ToolCalls,
			})
			a.mu.Unlock()
		}
		a.wg.Done()
	}()

	return nil
}
