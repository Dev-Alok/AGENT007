package orchestrator

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"agentic-coder/pkg/llm"
	"agentic-coder/pkg/tools"
)

// Agent manages the conversation state and implements the Thought->Action->Observation loop.
// Uses channels for concurrent tool execution to prevent blocking (Go concurrency pattern).
type Agent struct {
	registry      *tools.Registry
	llmClient     *llm.Client
	model         string
	contextWindow int
	maxIterations int

	mu           sync.RWMutex // Protects conversation state
	conversation []Message

	reflectionAttempts map[string]int // Track retry attempts per error pattern
}

// Message represents a chat message in the conversation history.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// NewAgent creates a new agent with the specified tool registry, LLM endpoint, and model.
// contextWindow limits token usage to prevent runaway costs (performance optimization).
func NewAgent(registry *tools.Registry, contextWindow int, llmEndpoint, model string) *Agent {
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClient(llmEndpoint), // Supports both Ollama and LM Studio
		model:              model,
		contextWindow:      contextWindow,
		maxIterations:      5, // Prevent infinite loops
		reflectionAttempts: make(map[string]int),
	}
}

// NewAgentWithAPIType creates a new agent with explicit API type selection.
// Use this when you need to specify OpenAI-compatible format (LM Studio) vs Ollama.
func NewAgentWithAPIType(registry *tools.Registry, contextWindow int, llmEndpoint, model string, apiType llm.APIType) *Agent {
	var client *llm.Client
	if apiType == llm.APIOpenAI {
		client = llm.NewClient(llmEndpoint) // Reuse NewClient with OpenAI mode
	} else {
		client = llm.NewClient(llmEndpoint)
	}

	return &Agent{
		registry:           registry,
		llmClient:          client,
		model:              model,
		contextWindow:      contextWindow,
		maxIterations:      50, // Prevent infinite loops
		reflectionAttempts: make(map[string]int),
	}
}

// SetModel allows changing the LLM model at runtime.
func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
}

// Execute runs the agent's main loop for a given task.
// Returns when task is complete or context is cancelled.
func (a *Agent) Execute(ctx context.Context, task string) (string, error) {
	// Initialize conversation with system prompt and user task
	systemPrompt := llm.BuildSystemPrompt(a.registry.List())

	a.mu.Lock()
	a.conversation = []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: task},
	}
	a.mu.Unlock()

	var finalOutput strings.Builder
	var iteration int

	for i := 0; i < a.maxIterations; i++ {
		select {
		case <-ctx.Done():
			return finalOutput.String(), ctx.Err()
		default:
		}

		iteration = i + 1

		// Step 1: Thought - Get LLM response
		response, err := a.getLLMResponse(ctx)
		if err != nil {
			return finalOutput.String(), fmt.Errorf("LLM error at iteration %d: %w", iteration, err)
		}

		// Check for completion signal
		if strings.Contains(response, "DONE_TASK_COMPLETE") {
			finalOutput.WriteString("\n=== Task Complete ===\n")
			break
		}

		// Step 2: Action - Parse and execute tool call
		toolCall, err := llm.ParseToolCall(response)
		if err != nil {
			// If not a tool call, treat as direct response
			a.mu.Lock()
			a.conversation = append(a.conversation, Message{Role: "assistant", Content: response})
			a.mu.Unlock()

			finalOutput.WriteString(fmt.Sprintf("\n[Iteration %d] Response:\n%s\n", iteration, response))
			continue
		}

		// Execute tool via goroutine with channel for non-blocking execution
		resultChan := make(chan string, 1)
		errorChan := make(chan error, 1)

		go func() {
			defer close(resultChan)
			defer close(errorChan)

			tool, ok := a.registry.Get(toolCall.Tool)
			if !ok {
				errorChan <- fmt.Errorf("unknown tool: %s", toolCall.Tool)
				return
			}

			result, err := tool.Execute(toolCall.Args)
			if err != nil {
				errorChan <- err
				return
			}

			resultChan <- result
		}()

		// Step 3: Observation - Wait for tool execution with timeout
		select {
		case <-ctx.Done():
			return finalOutput.String(), ctx.Err()
		case err := <-errorChan:
			// Reflection Loop: Analyze error and attempt correction
			a.mu.Lock()
			errorKey := fmt.Sprintf("%s:%v", toolCall.Tool, err)
			attempts := a.reflectionAttempts[errorKey]
			a.mu.Unlock()

			if attempts < 3 { // Max 3 retry attempts per error pattern
				// Add reflection message to conversation for learning
				reflectionMsg := fmt.Sprintf("Tool '%s' failed with error: %v. Attempting correction...\n", toolCall.Tool, err)

				a.mu.Lock()
				a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
				a.reflectionAttempts[errorKey] = attempts + 1
				a.mu.Unlock()

				finalOutput.WriteString(fmt.Sprintf("\n[Iteration %d] Tool '%s' failed: %v (attempt %d/3)\n",
					iteration, toolCall.Tool, err, attempts+1))

				// Continue loop for retry
				continue
			}

			return finalOutput.String(), fmt.Errorf("tool '%s' failed after 3 attempts: %w", toolCall.Tool, err)

		case result := <-resultChan:
			a.mu.Lock()
			a.conversation = append(a.conversation,
				Message{Role: "assistant", Content: response},
				Message{Role: "tool", Content: fmt.Sprintf("Tool '%s' executed successfully:\n%s", toolCall.Tool, result)},
			)
			a.reflectionAttempts = make(map[string]int) // Reset on success
			a.mu.Unlock()

			finalOutput.WriteString(fmt.Sprintf("\n[Iteration %d] Tool '%s' output:\n%s\n",
				iteration, toolCall.Tool, result))
		}
	}

	return finalOutput.String(), nil
}

// getLLMResponse sends the conversation to LLM and gets response.
// Implements context window management by truncating old messages (memory optimization).
func (a *Agent) getLLMResponse(ctx context.Context) (string, error) {
	a.mu.RLock()
	conversation := make([]Message, len(a.conversation))
	copy(conversation, a.conversation)
	a.mu.RUnlock()

	// Context window management: keep only recent messages to reduce token usage
	if len(conversation) > 10 { // Keep last ~10 exchanges
		conversation = conversation[len(conversation)-10:]
	}

	// Convert to LLM.Message format for API call
	llmMessages := make([]llm.Message, len(conversation))
	for i, msg := range conversation {
		llmMessages[i] = llm.Message{Role: msg.Role, Content: msg.Content}
	}

	response, err := a.llmClient.Chat(ctx, a.model, llmMessages, false) // Disable JSON mode for testing
	if err != nil {
		return "", fmt.Errorf("LLM chat failed: %w", err)
	}

	a.mu.Lock()
	a.conversation = append(a.conversation, Message{Role: "assistant", Content: response.Response})
	a.mu.Unlock()

	fmt.Fprintf(os.Stderr, "[DEBUG] LLM Response length: %d chars\n", len(response.Response))
	if len(response.Response) == 0 {
		fmt.Fprintf(os.Stderr, "[DEBUG] Empty response - checking conversation:\n")
		for i, msg := range llmMessages {
			fmt.Fprintf(os.Stderr, "  [%d] %s: %.50s...\n", i, msg.Role, msg.Content)
		}
	}
	return response.Response, nil
}

// GetConversation returns a copy of the current conversation history.
func (a *Agent) GetConversation() []Message {
	a.mu.RLock()
	defer a.mu.RUnlock()
	result := make([]Message, len(a.conversation))
	copy(result, a.conversation)
	return result
}

// Reset clears the conversation state.
func (a *Agent) Reset() {
	a.mu.Lock()
	a.conversation = nil
	a.reflectionAttempts = make(map[string]int)
	a.mu.Unlock()
}
