package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Client handles HTTP communication with Ollama.
type Client struct {
	baseURL       string
	apiKey        string
	temperature   float32
	numPredict    int
	topK          int
	topP          float32
	repeatPenalty float32
	seed          int
	httpClient    *http.Client
	mu            sync.RWMutex // Protects client state during reconnection
	tools         []Tool       // Native tools context optionally set
}

// ChatRequest represents the request format for Ollama chat endpoint.
type ChatRequest struct {
	Model    string                 `json:"model"`
	Messages []Message              `json:"messages"`
	Stream   bool                   `json:"stream"`
	Format   interface{}            `json:"format,omitempty"`  // Can be "json" or JSON schema object
	Options  map[string]interface{} `json:"options,omitempty"` // Model options like temperature
	Tools    []Tool                 `json:"tools,omitempty"`   // Native tool calling definitions
}

// Tool defines a tool available to the model.
type Tool struct {
	Type     string          `json:"type"` // always "function"
	Function ToolFunctionDef `json:"function"`
}

// ToolFunctionDef defines a specific function schema.
type ToolFunctionDef struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

// Parameters defines the JSON schema for function arguments.
type Parameters struct {
	Type       string                             `json:"type"` // usually "object"
	Properties map[string]ToolParameterDefinition `json:"properties"`
	Required   []string                           `json:"required"`
}

// ToolParameterDefinition defines argument types.
type ToolParameterDefinition struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// Message represents a chat message in Ollama format.
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // For assistant responses
	ToolName  string     `json:"tool_name,omitempty"`  // For role="tool"
}

// ChatResponse represents the response from Ollama.
type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`
	Usage     Usage     `json:"usage,omitempty"`
}

// Usage contains token usage statistics - Ollama format.
type Usage struct {
	PromptEvalCount int `json:"prompt_eval_count"`
	EvalCount       int `json:"eval_count"`
}

// ToolCall represents a structured tool call populated by the LLM natively.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the native name and arguments chosen by the LLM.
type ToolCallFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// NewClient creates a new client for Ollama API.
func NewClient(baseURL, apiKey string, temperature float32, numPredict, topK int, topP, repeatPenalty float32, seed int) *Client {
	return &Client{
		baseURL:       baseURL,
		apiKey:        apiKey,
		temperature:   temperature,
		numPredict:    numPredict,
		topK:          topK,
		topP:          topP,
		repeatPenalty: repeatPenalty,
		seed:          seed,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // Prevent indefinite hangs
		},
	}
}

// NewClientWithTimeout creates a new client with custom timeout.
func NewClientWithTimeout(baseURL, apiKey string, temperature float32, numPredict, topK int, topP, repeatPenalty float32, seed int, timeout time.Duration) *Client {
	return &Client{
		baseURL:       baseURL,
		apiKey:        apiKey,
		temperature:   temperature,
		numPredict:    numPredict,
		topK:          topK,
		topP:          topP,
		repeatPenalty: repeatPenalty,
		seed:          seed,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// HealthCheck verifies that the LLM server is running and accessible.
// Returns true if server is healthy, false otherwise.
func (c *Client) HealthCheck(ctx context.Context) error {
	c.mu.RLock()
	client := c.httpClient
	apiKey := c.apiKey
	c.mu.RUnlock()

	healthPath := "/api/tags"

	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+healthPath, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	return nil
}

// SetTools overrides the tools explicitly sent with each request.
func (c *Client) SetTools(tools []Tool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tools = tools
}

// RetryChat sends a message with retry logic and exponential backoff.
// Retries up to maxRetries times with increasing delays.
func (c *Client) RetryChat(ctx context.Context, model string, messages []Message, useJSONMode bool, maxRetries int) (*ChatResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s, 8s, etc.
			delay := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		response, err := c.Chat(ctx, model, messages, useJSONMode)
		if err == nil {
			return response, nil
		}

		lastErr = err

		// Don't retry on timeout or context cancellation
		if strings.Contains(err.Error(), "timeout") || strings.Contains(err.Error(), "context") {
			break
		}
	}

	return nil, fmt.Errorf("after %d attempts: %w", maxRetries+1, lastErr)
}

// Chat sends a message to the LLM and returns the response.
// Uses context for timeout/cancellation propagation (Go idiom).
func (c *Client) Chat(ctx context.Context, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	c.mu.RLock()
	client := c.httpClient
	c.mu.RUnlock()

	return c.chatOllama(ctx, client, model, messages, useJSONMode)
}

// chatOllama sends request to Ollama-compatible endpoint.
func (c *Client) chatOllama(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    c.tools,
		Stream:   false, // Non-streaming for simpler parsing
		Options: map[string]interface{}{
			"temperature":    c.temperature,
			"num_predict":    c.numPredict,
			"top_k":          c.topK,
			"top_p":          c.topP,
			"repeat_penalty": c.repeatPenalty,
			"seed":           c.seed,
		},
	}

	if useJSONMode {
		req.Format = "json"
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &chatResp, nil
}

// StreamChat sends a message and processes streaming responses.
// Uses channel-based streaming for non-blocking processing (Go concurrency pattern).
func (c *Client) StreamChat(ctx context.Context, model string, messages []Message, useJSONMode bool, resultChan chan<- *ChatResponse, errChan chan<- error) {
	defer close(resultChan)
	defer close(errChan)

	c.mu.RLock()
	client := c.httpClient
	c.mu.RUnlock()

	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
		Tools:    c.tools,
		Options: map[string]interface{}{
			"temperature":    c.temperature,
			"num_predict":    c.numPredict,
			"top_k":          c.topK,
			"top_p":          c.topP,
			"repeat_penalty": c.repeatPenalty,
			"seed":           c.seed,
		},
	}

	if useJSONMode {
		req.Format = "json"
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		errChan <- fmt.Errorf("failed to marshal request: %w", err)
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", strings.NewReader(string(jsonData)))
	if err != nil {
		errChan <- fmt.Errorf("failed to create HTTP request: %w", err)
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		errChan <- fmt.Errorf("HTTP request failed: %w", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		errChan <- fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	var fullResponse strings.Builder

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			errChan <- ctx.Err()
			return
		default:
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ChatResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			errChan <- fmt.Errorf("failed to parse stream chunk: %w", err)
			return
		}

		if len(chunk.Message.ToolCalls) > 0 {
			log.Printf("DEBUG: Received tool call inside stream! %+v\n", chunk.Message.ToolCalls)
		}

		fullResponse.WriteString(chunk.Message.Content)

		// Note: For Ollama, ToolCalls are typically generated all at once in the final chunk
		// or fragmented. We copy any we receive directly to the channel.
		resultChan <- &ChatResponse{
			Model:     chunk.Model,
			CreatedAt: chunk.CreatedAt,
			Message: Message{
				Role:      "assistant",
				Content:   fullResponse.String(),
				ToolCalls: chunk.Message.ToolCalls,
			},
			Done:  chunk.Done,
			Usage: chunk.Usage,
		}

		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		errChan <- fmt.Errorf("stream reading error: %w", err)
		return
	}
}

// ExtractToolCalls retrieves the safe deserialized native tool structs from the final model response chunk.
func ExtractToolCalls(finalChunk *ChatResponse) ([]ToolCall, error) {
	if finalChunk == nil {
		return nil, fmt.Errorf("no tool calls generated by the model")
	}

	// 1. Check Native Tool Calls (Supported by Llama 3.1+, Mistral, Qwen, etc)
	if finalChunk.Message.ToolCalls != nil && len(finalChunk.Message.ToolCalls) > 0 {
		return finalChunk.Message.ToolCalls, nil
	}

	// 2. Fallback JSON Parsing (For models like deepseek, glm, or llama2 without native tools)
	content := finalChunk.Message.Content
	if strings.Contains(content, "\"tool\"") && strings.Contains(content, "{") {
		blocks := extractJSONBlocks(content)
		var calls []ToolCall
		for _, block := range blocks {
			var fallback struct {
				Tool string                 `json:"tool"`
				Args map[string]interface{} `json:"args"`
			}
			if err := json.Unmarshal([]byte(block), &fallback); err == nil && fallback.Tool != "" {
				calls = append(calls, ToolCall{
					Function: ToolCallFunction{
						Name:      fallback.Tool,
						Arguments: fallback.Args,
					},
				})
			}
		}
		if len(calls) > 0 {
			return calls, nil
		}
	}

	return nil, fmt.Errorf("no tool calls generated by the model")
}

// extractJSONBlocks finds all top-level JSON objects in a string using brace-depth tracking.
// This correctly handles nested objects like {"tool": "x", "args": {"path": "y"}}.
func extractJSONBlocks(text string) []string {
	var blocks []string
	depth := 0
	start := -1

	for i, ch := range text {
		switch ch {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			if depth > 0 {
				depth--
				if depth == 0 && start >= 0 {
					blocks = append(blocks, text[start:i+1])
					start = -1
				}
			}
		}
	}

	return blocks
}

// BuildSystemPrompt creates an intelligent system prompt.
// Minimal prompt engineering is needed now because we use explicit structural tool schemas.
func BuildSystemPrompt(availableTools []Tool) string {
	var sb strings.Builder
	sb.WriteString(`You are an autonomous AI coding agent pair-programming with the user.

Your workflow:
1. THINK about what you need to do step by step to solve the user's objective.
2. Observe your available tools, and call exactly ONE tool to gather context or execute an action.
3. Wait to observe the result of the tool call.
4. Repeat this chain until the task is complete.

AVAILABLE TOOLS:
`)

	for _, t := range availableTools {
		schemaBytes, _ := json.MarshalIndent(t.Function, "", "  ")
		sb.WriteString(string(schemaBytes))
		sb.WriteString("\n")
	}

	sb.WriteString(`
If the model requires manual tool format usage, respond with ONLY a single valid JSON object representing your tool call:
{"tool": "tool_name", "args": {"arg1": "value1"}}

CRITICAL RULES:
- Use absolute or relative paths from the current working directory for all file operations.
- Always perform targeted 'grep_search' queries instead of dumping whole codebases.
- Your thinking process should be outputted within <think> tags.
`)
	return sb.String()
}
