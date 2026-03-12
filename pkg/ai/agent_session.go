package ai

import (
	"context"
	"fmt"
)

type AgentSessionOption[T any] func(*AgentSession[T])

// AgentSession manages a continuous chat session, tracking the underlying ChatRequest
// and providing a single global channel for responses of type T.
type AgentSession[T any] struct {
	Client      ChatClientInterface
	Req         *ChatRequest
	GlobalChan  chan T
	Transformer func(context.Context, <-chan *ChatResponse) <-chan T
	ctx         context.Context
	cancel      context.CancelFunc
}

// NewAgentSession creates a new AgentSession with the given client and base request.
// Use options like WithTransformer or WithAccumulator to specify the output type.
func NewAgentSession[T any](ctx context.Context, client ChatClientInterface, req *ChatRequest, opts ...AgentSessionOption[T]) *AgentSession[T] {
	ctx, cancel := context.WithCancel(ctx)
	session := &AgentSession[T]{
		Client: client,
		Req:    req,
		// We can't pre-allocate GlobalChan with T if we don't know the capacity needed,
		// but 100 is a safe default for streaming.
		GlobalChan: make(chan T, 100),
		ctx:        ctx,
		cancel:     cancel,
	}
	for _, opt := range opts {
		opt(session)
	}
	return session
}

// WithTransformer provides an option to add a channel transformer to the AgentSession.
func WithTransformer[T any](transformer func(context.Context, <-chan *ChatResponse) <-chan T) AgentSessionOption[T] {
	return func(a *AgentSession[T]) {
		a.Transformer = transformer
	}
}

// WithAccumulator provides an option to use the StreamAccumulator as a channel transformer.
// This results in the Session emitting *AccumulatedResponse.
func WithAccumulator() AgentSessionOption[*AccumulatedResponse] {
	return func(a *AgentSession[*AccumulatedResponse]) {
		a.Transformer = func(ctx context.Context, ch <-chan *ChatResponse) <-chan *AccumulatedResponse {
			return StreamAccumulator(ctx, ch, true)
		}
	}
}

// Recv returns a receive-only channel for all chat responses across multiple turns.
func (a *AgentSession[T]) Recv() <-chan T {
	return a.GlobalChan
}

// RecvAny returns a receive-only channel of any, wrapping the internal typed channel.
// This is useful for registry management where the session type is unknown.
func (a *AgentSession[T]) RecvAny() <-chan any {
	ch := make(chan any, cap(a.GlobalChan))
	go func() {
		for v := range a.GlobalChan {
			ch <- v
		}
		close(ch)
	}()
	return ch
}

// Stop cancels the session context and closes the global channel.
func (a *AgentSession[T]) Stop() {
	a.cancel()
	close(a.GlobalChan)
}


// SendUserMessage appends a user message to the request and triggers a streaming chat.
func (a *AgentSession[T]) SendUserMessage(ctx context.Context, msg string) error {
	a.Req.AddMessage(MessageRoleUser, msg)
	return a.streamChat(ctx)
}

// SendMessages appends one or more pre-constructed messages (like tool results) and triggers a streaming chat.
func (a *AgentSession[T]) SendMessages(ctx context.Context, msgs ...Message) error {
	a.Req.Messages = append(a.Req.Messages, msgs...)
	return a.streamChat(ctx)
}

// streamChat performs the streamed chat request and pipes the results into GlobalChan.
func (a *AgentSession[T]) streamChat(ctx context.Context) error {
	ch := make(chan *ChatResponse)

	// Start the request in a goroutine
	go func() {
		if err := a.Client.ChatStreamed(ctx, *a.Req, ch); err != nil {
			fmt.Printf("ChatStreamed error: %v\n", err)
		}
	}()

	// Pipe results to internal accumulator AND user output channel.
	go func() {
		// Channels for fan-out
		historyCh := make(chan *ChatResponse, 100)
		outputSrcCh := make(chan *ChatResponse, 100)

		// Start internal accumulator for history
		accCh := StreamAccumulator(ctx, historyCh, false)
		var finalRes *AccumulatedResponse
		accDone := make(chan struct{})
		go func() {
			for res := range accCh {
				if res != nil {
					finalRes = res
				}
			}
			close(accDone)
		}()

		// Prepare the user-facing output channel.
		var output <-chan T
		if a.Transformer != nil {
			output = a.Transformer(a.ctx, outputSrcCh)
		} else {
			outputDefault := (<-chan *ChatResponse)(outputSrcCh)
			output = any(outputDefault).(<-chan T)
		}

		// Distributor loop
		go func() {
			defer close(historyCh)
			defer close(outputSrcCh)
			for resp := range ch {
				select {
				case historyCh <- resp:
				case <-a.ctx.Done():
					return
				case <-ctx.Done():
					return
				}
				select {
				case outputSrcCh <- resp:
				case <-a.ctx.Done():
					return
				case <-ctx.Done():
					return
				}
			}
		}()

		// Pipe the results from 'output' into the global session channel.
		for res := range output {
			select {
			case a.GlobalChan <- res:
			case <-a.ctx.Done():
				return
			case <-ctx.Done():
				return
			}
		}

		// Wait for internal history accumulation to finish
		<-accDone

		// When the stream is done, append the assistant's final accumulated response to the request
		if finalRes != nil && finalRes.Chunk != nil && finalRes.Chunk.Done {
			msg := Message{
				Role:             MessageRoleAssistant,
				Content:          finalRes.Content,
				ReasoningContent: finalRes.ReasoningContent,
			}
			if len(finalRes.ToolCalls) > 0 {
				msg.ToolCalls = finalRes.ToolCalls
			}
			a.Req.Messages = append(a.Req.Messages, msg)
		}
	}()

	return nil
}
