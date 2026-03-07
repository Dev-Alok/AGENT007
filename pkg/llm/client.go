package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Client handles HTTP communication with Ollama or OpenAI-compatible APIs (LM Studio).
// Implements raw JSON-RPC/HTTP logic for maximum control (no heavy wrappers).
type Client struct {
	baseURL    string
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
// Using explicit structs instead of json.RawMessage for type safety.
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
	Type string `json:"type"` // "json_object"
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
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		apiType: APIOllama,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // Prevent indefinite hangs
		},
	}
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
		req.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}

	jsonData, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	fmt.Fprintf(os.Stderr, "[DEBUG] OpenAI Request:\n%s\n", string(jsonData))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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
		fmt.Fprintf(os.Stderr, "[DEBUG] HTTP Status: %d\n", resp.StatusCode)
		fmt.Fprintf(os.Stderr, "[DEBUG] Response body: %s\n", string(body))
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var openAIResp OpenAIChatResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		fmt.Fprintf(os.Stderr, "[DEBUG] JSON parse error: %v\n", err)
		fmt.Fprintf(os.Stderr, "[DEBUG] Raw body: %s\n", string(body))
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	fmt.Fprintf(os.Stderr, "[DEBUG] Parsed response - Choices: %d, Model: %s\n", len(openAIResp.Choices), openAIResp.Model)

	if len(openAIResp.Choices) == 0 {
		fmt.Fprintf(os.Stderr, "[DEBUG] No choices in response\n")
		return nil, fmt.Errorf("empty response from API")
	}

	content := openAIResp.Choices[0].Message.Content
	fmt.Fprintf(os.Stderr, "[DEBUG] Content length: %d chars\n", len(content))

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

// ParseToolCall attempts to parse a tool call from LLM text output.
// Uses JSON parsing first, falls back to regex for unstructured responses.
func ParseToolCall(text string) (*ToolCall, error) {
	text = strings.TrimSpace(text)

	// Try direct JSON parsing (preferred when using JSON mode)
	var toolCall ToolCall
	if err := json.Unmarshal([]byte(text), &toolCall); err == nil {
		if toolCall.Tool != "" {
			return &toolCall, nil
		}
	}

	// Fallback: Try to extract JSON object from text
	jsonStart := strings.Index(text, "{")
	jsonEnd := strings.LastIndex(text, "}")
	if jsonStart >= 0 && jsonEnd > jsonStart {
		jsonStr := text[jsonStart : jsonEnd+1]
		var toolCall ToolCall
		if err := json.Unmarshal([]byte(jsonStr), &toolCall); err == nil {
			if toolCall.Tool != "" {
				return &toolCall, nil
			}
		}
	}

	return nil, fmt.Errorf("no valid tool call found in response: %s", text)
}

// BuildSystemPrompt creates a system prompt that enforces structured output.
// Critical for getting reliable JSON responses from the LLM.
func BuildSystemPrompt(tools []string) string {
	toolList := strings.Join(tools, ", ")
	return fmt.Sprintf("You are an autonomous coding agent with access to the following tools: %s.\n\nWhen you need to use a tool, respond ONLY with a valid JSON object in this exact format:\n{\"tool\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}\n\nRules:\n1. NEVER include markdown formatting (no ```json blocks)\n2. NEVER add explanations before or after the JSON\n3. ALWAYS use valid JSON syntax with double quotes\n4. If you need to call a tool, return ONLY the JSON object\n5. When your task is complete and no further tools are needed, respond with DONE_TASK_COMPLETE\n\nAvailable tools: %s", toolList, toolList)
}
