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

type ProviderType string

const (
	ProviderLMStudio ProviderType = "lmstudio"
	ProviderOpenAI   ProviderType = "openai"
)

type Client struct {
	baseURL       string
	apiKey        string
	provider      ProviderType
	temperature   float32
	numPredict    int
	topK          int
	topP          float32
	repeatPenalty float32
	seed          int
	httpClient    *http.Client
	mu            sync.RWMutex
	tools         []Tool
}

type ChatRequest struct {
	Model    string                 `json:"model"`
	Messages []Message              `json:"messages"`
	Stream   bool                   `json:"stream"`
	Format   interface{}            `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Tools    []Tool                 `json:"tools,omitempty"`
}

type LMStudioChatRequest struct {
	Model    string                 `json:"model"`
	Messages []Message              `json:"messages"`
	Stream   bool                   `json:"stream"`
	Format   interface{}            `json:"format,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Tools    []Tool                 `json:"tools,omitempty"`
}

type Tool struct {
	Type     string          `json:"type"`
	Function ToolFunctionDef `json:"function"`
}

type ToolFunctionDef struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type Parameters struct {
	Type       string                             `json:"type"`
	Properties map[string]ToolParameterDefinition `json:"properties"`
	Required   []string                           `json:"required"`
}

type ToolParameterDefinition struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
}

type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`
	Usage     Usage     `json:"usage,omitempty"`
}

type Usage struct {
	PromptEvalCount int `json:"prompt_eval_count"`
	EvalCount       int `json:"eval_count"`
}

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

func NewClient(baseURL, apiKey string, provider ProviderType, temperature float32, numPredict, topK int, topP, repeatPenalty float32, seed int) *Client {
	return &Client{
		baseURL:       strings.TrimSuffix(baseURL, "/"),
		apiKey:        apiKey,
		provider:      provider,
		temperature:   temperature,
		numPredict:    numPredict,
		topK:          topK,
		topP:          topP,
		repeatPenalty: repeatPenalty,
		seed:          seed,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func NewClientWithTimeout(baseURL, apiKey string, provider ProviderType, temperature float32, numPredict, topK int, topP, repeatPenalty float32, seed int, timeout time.Duration) *Client {
	return &Client{
		baseURL:       strings.TrimSuffix(baseURL, "/"),
		apiKey:        apiKey,
		provider:      provider,
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

func (c *Client) HealthCheck(ctx context.Context) error {
	c.mu.RLock()
	client := c.httpClient
	apiKey := c.apiKey
	c.mu.RUnlock()

	var healthPath string
	if c.provider == ProviderLMStudio {
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

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	return nil
}

func (c *Client) SetTools(tools []Tool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tools = tools
}

func (c *Client) RetryChat(ctx context.Context, model string, messages []Message, useJSONMode bool, maxRetries int) (*ChatResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
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

		if strings.Contains(err.Error(), "timeout") || strings.Contains(err.Error(), "context") {
			break
		}
	}

	return nil, fmt.Errorf("after %d attempts: %w", maxRetries+1, lastErr)
}

func (c *Client) Chat(ctx context.Context, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	c.mu.RLock()
	client := c.httpClient
	c.mu.RUnlock()

	if c.provider == ProviderLMStudio {
		return c.chatLMStudio(ctx, client, model, messages, useJSONMode)
	}

	return c.chatOllama(ctx, client, model, messages, useJSONMode)
}

func (c *Client) chatLMStudio(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	return c.sendRequest(ctx, client, model, messages, useJSONMode, "/v1/chat/completions")
}

func (c *Client) chatOllama(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool) (*ChatResponse, error) {
	return c.sendRequest(ctx, client, model, messages, useJSONMode, "/api/chat")
}

func (c *Client) sendRequest(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool, endpointPath string) (*ChatResponse, error) {
	req := c.buildChatRequest(model, messages, useJSONMode)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+endpointPath, strings.NewReader(string(jsonData)))
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

func (c *Client) buildChatRequest(model string, messages []Message, useJSONMode bool) interface{} {
	if c.provider == ProviderLMStudio {
		req := LMStudioChatRequest{
			Model:    model,
			Messages: messages,
			Tools:    c.tools,
			Stream:   false,
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
		return req
	}

	req := ChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    c.tools,
		Stream:   false,
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
	return req
}

func (c *Client) StreamChat(ctx context.Context, model string, messages []Message, useJSONMode bool, resultChan chan<- *ChatResponse, errChan chan<- error) {
	defer close(resultChan)
	defer close(errChan)

	c.mu.RLock()
	client := c.httpClient
	c.mu.RUnlock()

	if c.provider == ProviderLMStudio {
		c.streamLMStudio(ctx, client, model, messages, useJSONMode, resultChan, errChan)
	} else {
		c.streamOllama(ctx, client, model, messages, useJSONMode, resultChan, errChan)
	}
}

func (c *Client) streamLMStudio(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool, resultChan chan<- *ChatResponse, errChan chan<- error) {
	req := LMStudioChatRequest{
		Model:    model,
		Messages: messages,
		Tools:    c.tools,
		Stream:   true,
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

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", strings.NewReader(string(jsonData)))
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

		lineStr := strings.TrimSpace(string(line))
		if lineStr == "data: [DONE]" || lineStr == "[DONE]" {
			break
		}

		if strings.HasPrefix(lineStr, "data: ") {
			lineStr = lineStr[6:]
		}

		var chunk ChatResponse
		if err := json.Unmarshal([]byte(lineStr), &chunk); err != nil {
			log.Printf("Warning: failed to parse stream chunk: %v", err)
			continue
		}

		if len(chunk.Message.ToolCalls) > 0 {
			log.Printf("DEBUG: Received tool call inside stream! %+v\n", chunk.Message.ToolCalls)
		}

		fullResponse.WriteString(chunk.Message.Content)

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

func (c *Client) streamOllama(ctx context.Context, client *http.Client, model string, messages []Message, useJSONMode bool, resultChan chan<- *ChatResponse, errChan chan<- error) {
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

func ExtractToolCalls(finalChunk *ChatResponse) ([]ToolCall, error) {
	if finalChunk == nil {
		return nil, fmt.Errorf("no tool calls generated by the model")
	}

	if finalChunk.Message.ToolCalls != nil && len(finalChunk.Message.ToolCalls) > 0 {
		return finalChunk.Message.ToolCalls, nil
	}

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
