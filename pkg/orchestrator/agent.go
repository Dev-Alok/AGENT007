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

type EventType string

const (
	EventThinking   EventType = "thinking"
	EventPlanning   EventType = "planning"
	EventToolCall   EventType = "tool_call"
	EventToolResult EventType = "tool_result"
	EventToolError  EventType = "tool_error"
	EventIteration  EventType = "iteration"
	EventComplete   EventType = "complete"
	EventStream     EventType = "stream"
	EventRetry      EventType = "retry"
	EventStatus     EventType = "status"
)

type AgentEvent struct {
	Type      EventType       `json:"type"`
	Message   string          `json:"message"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
	Timestamp time.Time       `json:"timestamp"`
}

type ToolCallMeta struct {
	ToolName string                 `json:"tool_name"`
	Args     map[string]interface{} `json:"args,omitempty"`
}

type IterationMeta struct {
	Current int `json:"current"`
	Max     int `json:"max"`
}

type RetryMeta struct {
	ToolName    string `json:"tool_name"`
	Attempt     int    `json:"attempt"`
	MaxAttempts int    `json:"max_attempts"`
	Error       string `json:"error"`
}

type StreamCallback func(content string)

type EventCallback func(event AgentEvent)

type Agent struct {
	registry      *tools.Registry
	llmClient     *llm.Client
	model         string
	contextWindow int
	maxIterations int

	mu           sync.RWMutex
	conversation []Message

	reflectionAttempts map[string]int
	streamCallback     StreamCallback
	eventCallback      EventCallback
}

type Message struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []llm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolName   string         `json:"tool_name,omitempty"`
}

func NewAgent(registry *tools.Registry, cfg *config.AgentConfig) *Agent {
	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = 20
	}
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClient(cfg.LLMEndpoint, cfg.APIKey, llm.ProviderType(cfg.Provider), cfg.Temperature, cfg.NumPredict, cfg.TopK, cfg.TopP, cfg.RepeatPenalty, cfg.Seed),
		model:              cfg.Model,
		contextWindow:      cfg.ContextWindow,
		maxIterations:      maxIter,
		reflectionAttempts: make(map[string]int),
	}
}

func NewAgentWithTimeout(registry *tools.Registry, cfg *config.AgentConfig, timeout time.Duration) *Agent {
	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = 20
	}
	return &Agent{
		registry:           registry,
		llmClient:          llm.NewClientWithTimeout(cfg.LLMEndpoint, cfg.APIKey, llm.ProviderType(cfg.Provider), cfg.Temperature, cfg.NumPredict, cfg.TopK, cfg.TopP, cfg.RepeatPenalty, cfg.Seed, timeout),
		model:              cfg.Model,
		contextWindow:      cfg.ContextWindow,
		maxIterations:      maxIter,
		reflectionAttempts: make(map[string]int),
	}
}

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
}

func (a *Agent) SetStreamCallback(callback StreamCallback) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.streamCallback = callback
}

func (a *Agent) SetEventCallback(callback EventCallback) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.eventCallback = callback
}

func (a *Agent) emit(event AgentEvent) {
	a.mu.RLock()
	callback := a.eventCallback
	a.mu.RUnlock()
	if callback != nil {
		callback(event)
	}
}

func (a *Agent) emitWithMeta(eventType EventType, message string, meta interface{}) {
	var metaBytes []byte
	if meta != nil {
		metaBytes, _ = json.Marshal(meta)
	}
	a.emit(AgentEvent{
		Type:      eventType,
		Message:   message,
		Metadata:  metaBytes,
		Timestamp: time.Now(),
	})
}

func (a *Agent) stream(content string) {
	a.mu.RLock()
	callback := a.streamCallback
	a.mu.RUnlock()
	if callback != nil {
		callback(content)
	}
}

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

func (a *Agent) Execute(ctx context.Context, task string) (string, error) {
	nativeTools := a.getNativeTools()

	systemPrompt := llm.BuildSystemPrompt(nativeTools)

	cwd, err := os.Getwd()
	if err == nil {
		systemPrompt += fmt.Sprintf("\n\nCurrent working directory: %s", cwd)
	}

	a.mu.Lock()
	a.conversation = []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: task},
	}
	a.mu.Unlock()

	a.emitWithMeta(EventStatus, "Starting agent execution...", IterationMeta{Current: 0, Max: a.maxIterations})

	var finalOutput strings.Builder
	var iteration int
	var toolsUsed bool

	for i := 0; i < a.maxIterations; i++ {
		select {
		case <-ctx.Done():
			a.emitWithMeta(EventStatus, "Execution cancelled", nil)
			return finalOutput.String(), ctx.Err()
		default:
		}

		iteration = i + 1
		a.emitWithMeta(EventIteration, fmt.Sprintf("Iteration %d/%d", iteration, a.maxIterations), IterationMeta{Current: iteration, Max: a.maxIterations})

		response, toolCalls, err := a.getLLMResponseStream(ctx, &finalOutput, a.streamCallback)
		if err != nil {
			a.emitWithMeta(EventStatus, fmt.Sprintf("LLM error at iteration %d: %v", iteration, err), nil)
			return finalOutput.String(), fmt.Errorf("LLM error at iteration %d: %w", iteration, err)
		}

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

		if len(toolCalls) == 0 {
			a.mu.Lock()
			a.reflectionAttempts = make(map[string]int)
			a.mu.Unlock()

			if !toolsUsed {
				a.emitWithMeta(EventComplete, "Task completed successfully", nil)
			} else {
				a.emitWithMeta(EventComplete, "Task completed successfully", nil)
			}
			break
		}

		if !toolsUsed {
			toolsUsed = true
		}

		a.emitWithMeta(EventIteration, fmt.Sprintf("Iteration %d/%d", iteration, a.maxIterations), IterationMeta{Current: iteration, Max: a.maxIterations})
		a.emitWithMeta(EventPlanning, fmt.Sprintf("Executing %d tool(s): %s", len(toolCalls), formatToolNames(toolCalls)), nil)

		var wg sync.WaitGroup
		var toolMu sync.Mutex

		for _, tc := range toolCalls {
			wg.Add(1)
			go func(toolCall llm.ToolCall) {
				defer wg.Done()

				tool, ok := a.registry.Get(toolCall.Function.Name)
				if !ok {
					errMsg := fmt.Sprintf("unknown tool: %s", toolCall.Function.Name)
					a.emitWithMeta(EventToolError, fmt.Sprintf("Tool '%s' not found", toolCall.Function.Name), ToolCallMeta{ToolName: toolCall.Function.Name})

					a.mu.Lock()
					errorKey := fmt.Sprintf("%s:%v", toolCall.Function.Name, errMsg)
					attempts := a.reflectionAttempts[errorKey]

					if attempts < 3 {
						reflectionMsg := fmt.Sprintf("Tool '%s' failed: %s. Call a different tool.", toolCall.Function.Name, errMsg)
						a.conversation = append(a.conversation, Message{Role: "system", Content: reflectionMsg})
						a.reflectionAttempts[errorKey] = attempts + 1
						a.emitWithMeta(EventRetry, fmt.Sprintf("Retrying with alternative tool (attempt %d/3)", attempts+1), RetryMeta{ToolName: toolCall.Function.Name, Attempt: attempts + 1, MaxAttempts: 3, Error: errMsg})
					}
					a.mu.Unlock()

					toolMu.Lock()
					msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v\n", toolCall.Function.Name, errMsg)
					finalOutput.WriteString(msg)
					a.stream(msg)
					toolMu.Unlock()
					return
				}

				args, _ := toolCall.Function.ParseArguments()
				a.emitWithMeta(EventToolCall, fmt.Sprintf("Calling %s...", toolCall.Function.Name), ToolCallMeta{ToolName: toolCall.Function.Name, Args: args})

				result, err := tool.Execute(ctx, args)

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

					a.emitWithMeta(EventToolError, fmt.Sprintf("Tool '%s' failed: %v (attempt %d/3)", toolCall.Function.Name, err, attempts+1), RetryMeta{ToolName: toolCall.Function.Name, Attempt: attempts + 1, MaxAttempts: 3, Error: err.Error()})

					toolMu.Lock()
					msg := fmt.Sprintf("\n⚠️ Tool '%s' failed: %v (attempt %d/3)\n", toolCall.Function.Name, err, attempts+1)
					finalOutput.WriteString(msg)
					a.stream(msg)
					toolMu.Unlock()
					return
				}

				toolMu.Lock()

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

				a.emitWithMeta(EventToolResult, dynamicMsg, ToolCallMeta{ToolName: toolCall.Function.Name, Args: args})

				switch toolCall.Function.Name {
				case "run_command", "grep_search":
					msg := fmt.Sprintf("\n%s\n", result)
					finalOutput.WriteString(msg)
					a.stream(msg)
				}

				toolMu.Unlock()

				a.mu.Lock()
				a.conversation = append(a.conversation,
					Message{Role: "tool", Content: result, ToolCallID: toolCall.ID, ToolName: toolCall.Function.Name},
				)
				a.reflectionAttempts = make(map[string]int)
				a.mu.Unlock()

			}(tc)
		}

		wg.Wait()

		a.emitWithMeta(EventThinking, "Analyzing tool results...", nil)
		continue
	}

	return finalOutput.String(), nil
}

func formatToolNames(toolCalls []llm.ToolCall) string {
	names := make([]string, len(toolCalls))
	for i, tc := range toolCalls {
		names[i] = tc.Function.Name
	}
	return strings.Join(names, ", ")
}

func (a *Agent) getLLMResponseStream(ctx context.Context, finalOutput *strings.Builder, callback StreamCallback) (string, []llm.ToolCall, error) {
	a.mu.RLock()
	conversation := make([]Message, len(a.conversation))
	copy(conversation, a.conversation)
	a.mu.RUnlock()

	maxChars := a.contextWindow * 4
	totalChars := 0
	for _, msg := range conversation {
		totalChars += len(msg.Content)
	}
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

	resultChan := make(chan *llm.ChatResponse, 100)
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

			newContent := ""
			newThinking := ""
			if resp.Message.Content != "" {
				newContent = strings.TrimPrefix(resp.Message.Content, fullResponse.String())
			}
			if resp.Message.ReasoningContent != "" {
				newThinking = resp.Message.ReasoningContent
			}

			if newThinking != "" {
				fullResponse.WriteString(newThinking)
				finalOutput.WriteString(newThinking)
				if callback != nil {
					callback(newThinking)
				}
				a.emitWithMeta(EventThinking, "Reasoning: "+newThinking, nil)
			}

			if newContent != "" {
				fullResponse.WriteString(newContent)
				finalOutput.WriteString(newContent)
				if callback != nil {
					callback(newContent)
				}
			}
		case <-ctx.Done():
			return fullResponse.String(), nil, ctx.Err()
		}
	}
}

func (a *Agent) GetConversation() []Message {
	a.mu.RLock()
	defer a.mu.RUnlock()
	result := make([]Message, len(a.conversation))
	copy(result, a.conversation)
	return result
}

func (a *Agent) Reset() {
	a.mu.Lock()
	a.conversation = nil
	a.reflectionAttempts = make(map[string]int)
	a.mu.Unlock()
}

func (a *Agent) GetLLMClient() *llm.Client {
	return a.llmClient
}

func (a *Agent) SaveHistory(filepath string) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	data, err := json.MarshalIndent(a.conversation, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath, data, 0644)
}

func (a *Agent) LoadHistory(filepath string) error {
	data, err := os.ReadFile(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	return json.Unmarshal(data, &a.conversation)
}
