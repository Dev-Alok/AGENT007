package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"agentic-coder/pkg/config"
	"agentic-coder/pkg/llm"
	"agentic-coder/pkg/tools"
)

// Agent manages the conversation state and implements the Thought->Action->Observation loop.
type Agent struct {
	registry      *tools.Registry
	llmClient     *llm.Client
	model         string
	contextWindow int
	maxIterations int

	mu           sync.RWMutex
	conversation []Message

	reflectionAttempts map[string]int
}

// Message represents a chat message in the conversation history.
type Message struct {
	Role        string         `json:"role"`
	Content     string         `json:"content"`
	ToolCalls   []llm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID  string         `json:"tool_call_id,omitempty"`
	ToolName    string         `json:"tool_name,omitempty"`
}

// NewAgent creates a new agent with the specified tool registry and configuration.
func NewAgent(registry *tools.Registry, cfg *config.AgentConfig) *Agent {
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClient(cfg.LLMEndpoint, cfg.APIKey, llm.ProviderType(cfg.Provider), cfg.Temperature, cfg.NumPredict, cfg.TopK, cfg.TopP, cfg.RepeatPenalty, cfg.Seed),
		model:              cfg.Model,
		contextWindow:      cfg.ContextWindow,
		maxIterations:      5,
		reflectionAttempts: make(map[string]int),
	}
}

// NewAgentWithTimeout creates a new agent with custom timeout.
func NewAgentWithTimeout(registry *tools.Registry, cfg *config.AgentConfig, timeout time.Duration) *Agent {
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClientWithTimeout(cfg.LLMEndpoint, cfg.APIKey, llm.ProviderType(cfg.Provider), cfg.Temperature, cfg.NumPredict, cfg.TopK, cfg.TopP, cfg.RepeatPenalty, cfg.Seed, timeout),
		model:              cfg.Model,
		contextWindow:      cfg.ContextWindow,
		maxIterations:      5,
		reflectionAttempts: make(map[string]int),
	}
}

// SetModel allows changing the LLM model at runtime.
func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
}

// getNativeTools maps internal Registry tools to LLM-native tool schemas.
func (a *Agent) getNativeTools() []llm.Tool {
	var nativeTools []llm.Tool
	for _, toolName := range a.registry.List() {
		if t, ok := a.registry.Get(toolName); ok {
			props := make(map[string]llm.ToolParameterDefinition)
			for paramName, paramDef := range t.Parameters {
				props[paramName] = llm.ToolParameterDefinition{
					Type:        paramDef.Type,
					Description: paramDef.Description,
				}
			}

			nativeTools = append(nativeTools, llm.Tool{
				Type: "function",
				Function: llm.ToolFunctionDef{
					Name:        t.Name,
					Description: t.Description,
					Parameters: llm.Parameters{
						Type:       "object",
						Properties: props,
						Required:   t.Required,
					},
				},
			})
		}
	}
	return nativeTools
}

// Execute runs the agent's main loop for a given task.
// Returns when task is complete or context is cancelled.
// Each execution starts fresh with no conversation history.
func (a *Agent) Execute(ctx context.Context, task string) (string, error) {
	nativeTools := a.getNativeTools()

	// Initialize conversation with system prompt and user task (fresh each time)
	systemPrompt := llm.BuildSystemPrompt(nativeTools)

	// Add working directory context
	cwd, err := os.Getwd()
	if err == nil {
		systemPrompt += fmt.Sprintf("\n\nCurrent working directory: %s", cwd)
	}

	a.mu.Lock()
	// Always start fresh - no conversation history
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

		// Step 1: Thought - Get LLM response via streaming
		response, toolCalls, err := a.getLLMResponseStream(ctx, &finalOutput)
		if err != nil {
			return finalOutput.String(), fmt.Errorf("LLM error at iteration %d: %w", iteration, err)
		}

	// Append the assistant's unified response once before checking success states
		// Convert tool calls to have proper map arguments for serialization
		normalizedToolCalls := make([]llm.ToolCall, len(toolCalls))
		for i, tc := range toolCalls {
			normalizedToolCalls[i] = tc
			if args, err := tc.Function.ParseArguments(); err == nil && args != nil {
				normalizedToolCalls[i].Function.Arguments = args
			}
		}
		a.mu.Lock()
		a.conversation = append(a.conversation, Message{Role: "assistant", Content: response, ToolCalls: normalizedToolCalls})
		a.mu.Unlock()

		// If zero tool calls, the model decided it was done or wants to talk
		if len(toolCalls) == 0 {
			a.mu.Lock()
			a.reflectionAttempts = make(map[string]int) // Reset on exit
			a.mu.Unlock()

			break
		}
		// ... Parsing error logic handled inside getLLMResponseStream ...

		// Execute tools concurrently
		var wg sync.WaitGroup
		var toolMu sync.Mutex

		for _, tc := range toolCalls {
			wg.Add(1)
			go func(toolCall llm.ToolCall) {
				defer wg.Done()

				tool, ok := a.registry.Get(toolCall.Function.Name)
				if !ok {
					errMsg := fmt.Sprintf("unknown tool: %s", toolCall.Function.Name)
					a.mu.Lock()
					errorKey := fmt.Sprintf("%s:%v", toolCall.Function.Name, errMsg)
					attempts := a.reflectionAttempts[errorKey]

					if attempts < 3 { // Retry loop logic
						reflectionMsg := fmt.Sprintf("Tool '%s' failed: %s. Call a different tool.", toolCall.Function.Name, errMsg)
						a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
						a.reflectionAttempts[errorKey] = attempts + 1
					}
					a.mu.Unlock()

					toolMu.Lock()
					msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v\n", toolCall.Function.Name, errMsg)
					fmt.Fprint(os.Stdout, msg)
					finalOutput.WriteString(msg)
					toolMu.Unlock()
					return
				}

			// Execute tool
			args, _ := toolCall.Function.ParseArguments()
			result, err := tool.Execute(ctx, args)

				// Handle Error
				if err != nil {
					a.mu.Lock()
					errorKey := fmt.Sprintf("%s:%v", toolCall.Function.Name, err)
					attempts := a.reflectionAttempts[errorKey]

					if attempts < 3 {
						reflectionMsg := fmt.Sprintf("Tool '%s' failed with error: %v. Attempting correction. Provide corrected arguments.", toolCall.Function.Name, err)
						a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
						a.reflectionAttempts[errorKey] = attempts + 1
					}
					a.mu.Unlock()

				toolMu.Lock()
				msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v (attempt %d/3)\n", toolCall.Function.Name, err, attempts+1)
				fmt.Fprint(os.Stdout, msg)
				finalOutput.WriteString(msg)
				toolMu.Unlock()
				return
			}

				// Handle Success
				toolMu.Lock()

			// Generate dynamic context-aware message based on the tool executed
				var dynamicMsg string
				switch toolCall.Function.Name {
				case "list_directory":
					dynamicMsg = fmt.Sprintf("Listed directory: %v", args["path"])
				case "read_file":
					dynamicMsg = fmt.Sprintf("Read file: %v", args["path"])
				case "search_file":
					dynamicMsg = fmt.Sprintf("Searched for file: %v", args["name"])
				case "grep_search":
					dynamicMsg = fmt.Sprintf("Grep searched for \"%v\" in %v", args["query"], args["path"])
				case "write_file":
					dynamicMsg = fmt.Sprintf("Wrote to file: %v", args["path"])
				case "replace_file_content":
					dynamicMsg = fmt.Sprintf("Replaced content in file: %v", args["path"])
				case "read_url":
					dynamicMsg = fmt.Sprintf("Fetched URL: %v", args["url"])
				case "run_command":
					dynamicMsg = fmt.Sprintf("Ran command: %v", args["command"])
				default:
					dynamicMsg = fmt.Sprintf("Executed tool: %s", toolCall.Function.Name)
				}

			msg := fmt.Sprintf("\n✅ \033[32m%s\033[0m\n", dynamicMsg)
				fmt.Fprint(os.Stdout, msg)
				finalOutput.WriteString(msg)
				toolMu.Unlock()

			a.mu.Lock()
				a.conversation = append(a.conversation,
					Message{Role: "tool", Content: result, ToolCallID: toolCall.ID, ToolName: toolCall.Function.Name},
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

// getLLMResponseStream sends the conversation to LLM and processes the streamed response.
func (a *Agent) getLLMResponseStream(ctx context.Context, finalOutput *strings.Builder) (string, []llm.ToolCall, error) {
	a.mu.RLock()
	conversation := make([]Message, len(a.conversation))
	copy(conversation, a.conversation)
	a.mu.RUnlock()

	// Context window management
	maxChars := a.contextWindow * 4
	totalChars := 0
	for _, msg := range conversation {
		totalChars += len(msg.Content)
	}
	// Drop 2 older messages at a time (pair of user/assistant) to preserve alternating conversational state
	for totalChars > maxChars && len(conversation) > 3 {
		droppedMsg1 := conversation[1]
		droppedMsg2 := conversation[2]
		totalChars -= (len(droppedMsg1.Content) + len(droppedMsg2.Content))
		conversation = append(conversation[:1], conversation[3:]...)
	}

	llmMessages := make([]llm.Message, len(conversation))
	for i, msg := range conversation {
		llmMessages[i] = llm.Message{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCalls:  msg.ToolCalls,
			ToolCallID: msg.ToolCallID,
			ToolName:   msg.ToolName,
		}
	}

	nativeTools := a.getNativeTools()

	resultChan := make(chan *llm.ChatResponse, 100) // Buffer stream
	errChan := make(chan error, 1)

	a.llmClient.SetTools(nativeTools)
	go a.llmClient.StreamChat(ctx, a.model, llmMessages, false, resultChan, errChan)

	var fullResponse strings.Builder
	var finalChunk *llm.ChatResponse

	for {
		select {
		case err := <-errChan:
			if err != nil {
				return "", nil, fmt.Errorf("LLM stream failed: %w", err)
			}
			toolCalls, _ := llm.ExtractToolCalls(finalChunk)
			return fullResponse.String(), toolCalls, nil
		case resp, ok := <-resultChan:
			if !ok {
				toolCalls, _ := llm.ExtractToolCalls(finalChunk)
				return fullResponse.String(), toolCalls, nil
			}

			finalChunk = resp

		// Get content from response
		newContent := ""
		if resp.Message.Content != "" {
			newContent = strings.TrimPrefix(resp.Message.Content, fullResponse.String())
		}

		if newContent != "" {
			fullResponse.WriteString(newContent)
			finalOutput.WriteString(newContent)
		}

		// Stream content as it comes
			if newContent != "" {
				fmt.Fprint(os.Stdout, newContent)
			}
		case <-ctx.Done():
			return fullResponse.String(), nil, ctx.Err()
		}
	}
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
