package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Client handles HTTP communication with Ollama or OpenAI-compatible APIs (LM Studio).
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	mu         sync.RWMutex // Protects client state during reconnection
	apiType    APIType      // Ollama or OpenAI format
}

// APIType specifies which API format to use.
type APIType int

const (
	APIOllama APIType = iota
	APIOpenAI         // LM Studio uses OpenAI-compatible format
)

// ChatRequest represents the request format for Ollama chat endpoint.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Format   *Format   `json:"format,omitempty"` // For structured output
}

// OpenAIChatRequest represents the request format for OpenAI-compatible APIs.
type OpenAIChatRequest struct {
	Model          string          `json:"model"`
	Messages       []OpenAIMessage `json:"messages"`
	Stream         bool            `json:"stream"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"` // For JSON mode
}

// OpenAIMessage represents a chat message in OpenAI format.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Message represents a chat message in Ollama format.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Format specifies structured output format (JSON for tool calls) - Ollama format.
type Format struct {
	Type string `json:"type"` // "json"
}

// ResponseFormat specifies JSON response format - OpenAI/LM Studio format.
type ResponseFormat struct {
	Type string `json:"type"` // "json_schema"
}

// ChatResponse represents the response from Ollama.
type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`
	Done      bool      `json:"done"`
	Usage     Usage     `json:"usage,omitempty"`
}

// OpenAIChatResponse represents the response from OpenAI-compatible APIs.
type OpenAIChatResponse struct {
	Model   string      `json:"model"`
	Created int64       `json:"created"`
	Choices []Choice    `json:"choices"`
	Usage   OpenAIUsage `json:"usage,omitempty"`
}

// Choice represents a choice in the response.
type Choice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// Usage contains token usage statistics - Ollama format.
type Usage struct {
	PromptTokens   int `json:"prompt_tokens"`
	ResponseTokens int `json:"response_tokens"`
	TotalTokens    int `json:"total_tokens"`
}

// OpenAIUsage contains token usage statistics - OpenAI format.
type OpenAIUsage struct {
	PromptTokens   int `json:"prompt_tokens"`
	ResponseTokens int `json:"completion_tokens"`
	TotalTokens    int `json:"total_tokens"`
}

// ToolCall represents a structured tool call from the LLM.
// This is the output format we expect when using JSON mode.
type ToolCall struct {
	Tool string                 `json:"tool"`
	Args map[string]interface{} `json:"args"`
}

// NewClient creates a new client for Ollama API.
func NewClient(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		apiType: APIOllama,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // Prevent indefinite hangs
		},
	}
}

// NewClientWithAPIType creates a new client with specified API type.
func NewClientWithAPIType(baseURL, apiKey string, apiType APIType) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		apiType: apiType,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// NewClientWithTimeout creates a new client with custom timeout.
func NewClientWithTimeout(baseURL, apiKey string, timeout time.Duration) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		apiType: APIOllama,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// NewClientWithTimeoutAndAPIType creates a new client with custom timeout and API type.
func NewClientWithTimeoutAndAPIType(baseURL, apiKey string, apiType APIType, timeout time.Duration) *Client {
	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		apiType: apiType,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// SetAPIType allows changing the API type at runtime.
func (c *Client) SetAPIType(apiType APIType) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.apiType = apiType
}

// HealthCheck verifies that the LLM server is running and accessible.
// Returns true if server is healthy, false otherwise.
func (c *Client) HealthCheck(ctx context.Context) error {
	c.mu.RLock()
	client := c.httpClient
	apiType := c.apiType
	apiKey := c.apiKey
	c.mu.RUnlock()

	var healthPath string
	if apiType == APIOpenAI {
		healthPath = "/v1/models"
	} else {
		healthPath = "/api/tags"
	}

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
	apiType := c.apiType
	c.mu.RUnlock()

	if apiType == APIOpenAI {
		return c.chatOpenAI(ctx, client, model, messages, useJSONMode)
	}
	return c.chatOllama(ctx, client, model, messages, useJSONMode)
}

// chatOllama sends request to Ollama-compatible endpoint.
func (c *Client) chatOllama(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false, // Non-streaming for simpler parsing
	}

	if useJSONMode {
		format := Format{Type: "json"}
		req.Format = &format
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

// chatOpenAI sends request to OpenAI-compatible endpoint (LM Studio).
func (c *Client) chatOpenAI(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	// Convert Ollama Message format to OpenAI format
	openAIMessages := make([]OpenAIMessage, len(messages))
	for i, msg := range messages {
		openAIMessages[i] = OpenAIMessage{Role: msg.Role, Content: msg.Content}
	}

	req := OpenAIChatRequest{
		Model:    model,
		Messages: openAIMessages,
		Stream:   false,
	}

	if useJSONMode {
		// Note: The specific json_schema config may require defining the actual schema tree structure based on LM Studio version
		// but providing the type as json_schema fulfills the type requirement check initially.
		// Older/more robust LM studios might natively fall back to generic schema if omitted.
		req.ResponseFormat = &ResponseFormat{Type: "json_schema"}
	}

	jsonData, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", strings.NewReader(string(jsonData)))
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

	var openAIResp OpenAIChatResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(openAIResp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from API")
	}

	content := openAIResp.Choices[0].Message.Content

	// Convert OpenAI response to standard ChatResponse format
	return &ChatResponse{
		Model:    openAIResp.Model,
		Response: content,
		Done:     true, // OpenAI doesn't stream in this mode
		Usage: Usage{
			PromptTokens:   openAIResp.Usage.PromptTokens,
			ResponseTokens: openAIResp.Usage.ResponseTokens,
			TotalTokens:    openAIResp.Usage.TotalTokens,
		},
	}, nil
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
	}

	if useJSONMode {
		format := Format{Type: "json"}
		req.Format = &format
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

		fullResponse.WriteString(chunk.Response)

		// Send intermediate response for real-time feedback
		resultChan <- &ChatResponse{
			Model:     chunk.Model,
			CreatedAt: chunk.CreatedAt,
			Response:  fullResponse.String(),
			Done:      chunk.Done,
			Usage:     chunk.Usage,
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

// ParseToolCalls extracts all valid tool-call JSON objects from LLM text output.
// Uses brace-depth tracking instead of regex to correctly handle nested JSON (e.g. args with objects).
func ParseToolCalls(text string) ([]ToolCall, error) {
	text = strings.TrimSpace(text)
	var calls []ToolCall

	// Extract all top-level JSON objects using brace-depth tracking
	jsonBlocks := extractJSONBlocks(text)

	for _, block := range jsonBlocks {
		var tc ToolCall
		if err := json.Unmarshal([]byte(block), &tc); err == nil && tc.Tool != "" {
			calls = append(calls, tc)
		}
	}

	if len(calls) == 0 {
		return nil, fmt.Errorf("no valid tool call found in response")
	}

	return calls, nil
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

// BuildSystemPrompt creates an intelligent system prompt with chain-of-thought reasoning.
func BuildSystemPrompt(tools []string) string {
	toolList := strings.Join(tools, ", ")
	return fmt.Sprintf(`You are an autonomous AI coding agent. You have access to the following tools: %s.

Your workflow is:
1. THINK about what you need to do step by step.
2. Call ONE tool at a time to gather information or make changes.
3. OBSERVE the result of the tool call.
4. Repeat until the task is complete.

When you need to use a tool, respond with ONLY a single valid JSON object:
{"tool": "tool_name", "args": {"arg1": "value1"}}

CRITICAL RULES:
- Call only ONE tool per response. Wait for its result before calling the next.
- NEVER include explanations, markdown, or any text alongside the JSON.
- ALWAYS use valid JSON with double quotes.
- Use absolute or relative paths from the current working directory.
- When you are DONE and have gathered enough information to answer, respond with DONE_TASK_COMPLETE followed by your final answer on the next line.
- If the user asks a question that does NOT require tools, respond with DONE_TASK_COMPLETE followed by your answer.

Available tools: %s`, toolList, toolList)
}
