package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

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
func NewAgent(registry *tools.Registry, contextWindow int, llmEndpoint, apiKey, model string) *Agent {
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClient(llmEndpoint, apiKey), // Supports both Ollama and LM Studio
		model:              model,
		contextWindow:      contextWindow,
		maxIterations:      5, // Prevent infinite loops
		reflectionAttempts: make(map[string]int),
	}
}

// NewAgentWithTimeout creates a new agent with custom timeout.
func NewAgentWithTimeout(registry *tools.Registry, contextWindow int, llmEndpoint, apiKey, model string, timeout time.Duration) *Agent {
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClientWithTimeout(llmEndpoint, apiKey, timeout),
		model:              model,
		contextWindow:      contextWindow,
		maxIterations:      5, // Prevent infinite loops
		reflectionAttempts: make(map[string]int),
	}
}

// NewAgentWithAPIType creates a new agent with explicit API type selection.
// Use this when you need to specify OpenAI-compatible format (LM Studio) vs Ollama.
func NewAgentWithAPIType(registry *tools.Registry, contextWindow int, llmEndpoint, apiKey, model string, apiType llm.APIType) *Agent {
	client := llm.NewClientWithAPIType(llmEndpoint, apiKey, apiType)

	return &Agent{
		registry:           registry,
		llmClient:          client,
		model:              model,
		contextWindow:      contextWindow,
		maxIterations:      50, // Prevent infinite loops
		reflectionAttempts: make(map[string]int),
	}
}

// NewAgentWithTimeoutAndAPIType creates a new agent with custom timeout and API type.
func NewAgentWithTimeoutAndAPIType(registry *tools.Registry, contextWindow int, llmEndpoint, apiKey, model string, apiType llm.APIType, timeout time.Duration) *Agent {
	client := llm.NewClientWithTimeoutAndAPIType(llmEndpoint, apiKey, apiType, timeout)

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

	// Add working directory context
	cwd, err := os.Getwd()
	if err == nil {
		systemPrompt += fmt.Sprintf("\n\nCurrent working directory: %s", cwd)
	}

	a.mu.Lock()
	if len(a.conversation) == 0 {
		a.conversation = []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: task},
		}
	} else {
		// Update system prompt to maintain current working directory info
		if len(a.conversation) > 0 && a.conversation[0].Role == "system" {
			a.conversation[0].Content = systemPrompt
		}
		a.conversation = append(a.conversation, Message{Role: "user", Content: task})
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
			// Extract the final answer after DONE_TASK_COMPLETE
			finalAnswer := strings.TrimSpace(strings.SplitN(response, "DONE_TASK_COMPLETE", 2)[1])
			if finalAnswer != "" {
				msg := fmt.Sprintf("\n\033[36m🤖 Assistant:\033[0m\n%s\n", finalAnswer)
				fmt.Print(msg)
				finalOutput.WriteString(msg)
			}
			msg := "\n✨ Task Complete ✨\n"
			fmt.Print(msg)
			finalOutput.WriteString(msg)
			break
		}

		// Parse and execute tools
		toolCalls, err := llm.ParseToolCalls(response)
		if err != nil {
			// Track parsing errors specifically to give the LLM a chance to correct hallucinatory output
			a.mu.Lock()
			errorKey := "system:parsing_error"
			attempts := a.reflectionAttempts[errorKey]

			if attempts < 3 {
				// Retry loop for bad tool formatting
				reflectionMsg := fmt.Sprintf("Error: Your response was not a valid JSON tool call. %v. REMEMBER: You must respond with ONLY a single valid JSON object representing a tool call, and nothing else.", err)
				a.conversation = append(a.conversation, Message{Role: "assistant", Content: response})
				a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
				a.reflectionAttempts[errorKey] = attempts + 1
				a.mu.Unlock()

				msg := fmt.Sprintf("\n⚠️ LLM Format Error: %v. Retrying (attempt %d/3)...\n", err, attempts+1)
				fmt.Print(msg)
				finalOutput.WriteString(msg)
				continue // Let the LLM try again in the next iteration
			}

			// If max retries reached, treat as direct conversational response and break
			a.conversation = append(a.conversation, Message{Role: "assistant", Content: response})
			a.reflectionAttempts = make(map[string]int) // Reset on exit
			a.mu.Unlock()

			msg := fmt.Sprintf("\n\033[36m🤖 Assistant:\033[0m\n%s\n", response)
			fmt.Print(msg)
			finalOutput.WriteString(msg)
			break
		}

		// Append the assistant's unified response once before executing tools
		a.mu.Lock()
		a.conversation = append(a.conversation, Message{Role: "assistant", Content: response})
		a.mu.Unlock()

		// Execute tools concurrently
		var wg sync.WaitGroup
		var toolMu sync.Mutex

		for _, tc := range toolCalls {
			wg.Add(1)
			go func(toolCall llm.ToolCall) {
				defer wg.Done()

				tool, ok := a.registry.Get(toolCall.Tool)
				if !ok {
					errMsg := fmt.Sprintf("unknown tool: %s", toolCall.Tool)
					a.mu.Lock()
					errorKey := fmt.Sprintf("%s:%v", toolCall.Tool, errMsg)
					attempts := a.reflectionAttempts[errorKey]

					if attempts < 3 { // Retry loop logic
						reflectionMsg := fmt.Sprintf("Tool '%s' failed: %s. Attempting correction. REMEMBER: You must respond with ONLY a single valid JSON object. Do not output anything else.", toolCall.Tool, errMsg)
						a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
						a.reflectionAttempts[errorKey] = attempts + 1
					}
					a.mu.Unlock()

					toolMu.Lock()
					msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v\n", toolCall.Tool, errMsg)
					fmt.Print(msg)
					finalOutput.WriteString(msg)
					toolMu.Unlock()
					return
				}

				// Execute tool
				result, err := tool.Execute(ctx, toolCall.Args)

				// Handle Error
				if err != nil {
					a.mu.Lock()
					errorKey := fmt.Sprintf("%s:%v", toolCall.Tool, err)
					attempts := a.reflectionAttempts[errorKey]

					if attempts < 3 {
						reflectionMsg := fmt.Sprintf("Tool '%s' failed with error: %v. Attempting correction. REMEMBER: You must respond with ONLY a single valid JSON object. Do not output anything else.", toolCall.Tool, err)
						a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
						a.reflectionAttempts[errorKey] = attempts + 1
					}
					a.mu.Unlock()

					toolMu.Lock()
					msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v (attempt %d/3)\n", toolCall.Tool, err, attempts+1)
					fmt.Print(msg)
					finalOutput.WriteString(msg)
					toolMu.Unlock()
					return
				}

				// Handle Success
				toolMu.Lock()

				// Generate dynamic context-aware message based on the tool executed
				var dynamicMsg string
				switch toolCall.Tool {
				case "list_directory":
					dynamicMsg = fmt.Sprintf("Listed directory: %v", toolCall.Args["path"])
				case "read_file":
					dynamicMsg = fmt.Sprintf("Read file: %v", toolCall.Args["path"])
				case "search_file":
					dynamicMsg = fmt.Sprintf("Searched for file: %v", toolCall.Args["name"])
				case "grep_search":
					dynamicMsg = fmt.Sprintf("Grep searched for \"%v\" in %v", toolCall.Args["query"], toolCall.Args["path"])
				case "write_file":
					dynamicMsg = fmt.Sprintf("Wrote to file: %v", toolCall.Args["path"])
				case "replace_file_content":
					dynamicMsg = fmt.Sprintf("Replaced content in file: %v", toolCall.Args["path"])
				case "read_url":
					dynamicMsg = fmt.Sprintf("Fetched URL: %v", toolCall.Args["url"])
				case "run_command":
					dynamicMsg = fmt.Sprintf("Ran command: %v", toolCall.Args["command"])
				default:
					dynamicMsg = fmt.Sprintf("Executed tool: %s", toolCall.Tool)
				}

				msg := fmt.Sprintf("\n✅ \033[32m%s\033[0m\n", dynamicMsg)
				fmt.Print(msg)
				finalOutput.WriteString(msg)
				toolMu.Unlock()

				a.mu.Lock()
				a.conversation = append(a.conversation,
					Message{Role: "tool", Content: fmt.Sprintf("Tool '%s' executed successfully:\n%s", toolCall.Tool, result)},
				)
				a.reflectionAttempts = make(map[string]int) // Reset on success
				a.mu.Unlock()

			}(tc)
		}

		wg.Wait()

		// Let the loop continue to the next iteration so the LLM can analyze the tool results
		continue
	}

	return finalOutput.String(), nil
}

// getLLMResponse sends the conversation to LLM and gets response.
func (a *Agent) getLLMResponse(ctx context.Context) (string, error) {
	a.mu.RLock()
	conversation := make([]Message, len(a.conversation))
	copy(conversation, a.conversation)
	a.mu.RUnlock()

	// Context window management: Drop older messages if total length exceeds contextWindow limit
	// Approximation: 1 token ≈ 4 characters
	maxChars := a.contextWindow * 4
	totalChars := 0
	for _, msg := range conversation {
		totalChars += len(msg.Content)
	}
	// Drop older messages until we are under the limit
	for totalChars > maxChars && len(conversation) > 2 {
		droppedMsg := conversation[1]
		totalChars -= len(droppedMsg.Content)
		conversation = append(conversation[:1], conversation[2:]...)
	}

	// Convert to LLM.Message format for API call
	llmMessages := make([]llm.Message, len(conversation))
	for i, msg := range conversation {
		llmMessages[i] = llm.Message{Role: msg.Role, Content: msg.Content}
	}

	response, err := a.llmClient.Chat(ctx, a.model, llmMessages, false)
	if err != nil {
		return "", fmt.Errorf("LLM chat failed: %w", err)
	}

	// NOTE: Do NOT append assistant message here - the caller (Execute) handles conversation state
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

// GetLLMClient returns the underlying LLM client.
func (a *Agent) GetLLMClient() *llm.Client {
	return a.llmClient
}

// SaveHistory saves the current conversation to a JSON file.
func (a *Agent) SaveHistory(filepath string) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	data, err := json.MarshalIndent(a.conversation, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath, data, 0644)
}

// LoadHistory loads the conversation from a JSON file.
func (a *Agent) LoadHistory(filepath string) error {
	data, err := os.ReadFile(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // OK if file doesn't exist yet
		}
		return err
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	return json.Unmarshal(data, &a.conversation)
}
