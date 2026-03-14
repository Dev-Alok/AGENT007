package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type ProviderType string

const (
	ProviderLMStudio ProviderType = "lmstudio"
	ProviderOllama   ProviderType = "ollama"
)

// AgentConfig represents the configuration layout for the agent application.
// Used instead of CLI flags to support easy profile switching.
type AgentConfig struct {
	LLMEndpoint       string       `json:"llm_endpoint"`
	Model             string       `json:"model"`
	ContextWindow     int          `json:"context_window"`
	Temperature       float32      `json:"temperature"`
	NumPredict        int          `json:"num_predict"`
	TopK              int          `json:"top_k"`
	TopP              float32      `json:"top_p"`
	RepeatPenalty     float32      `json:"repeat_penalty"`
	Seed              int          `json:"seed"`
	LLMTimeoutSeconds int          `json:"llm_timeout_seconds"`
	Provider          ProviderType `json:"provider"`
	APIKey            string       `json:"api_key,omitempty"`
	MaxIterations     int          `json:"max_iterations,omitempty"` // Max tool call iterations per task
}

// LoadConfig parses a config.json file and returns an initialized AgentConfig.
// It applies safe defaults so the application can still boot if attributes are missing.
func LoadConfig(filePath string) (*AgentConfig, error) {
	cfg := &AgentConfig{
		LLMEndpoint:       "http://localhost:1234",
		Model:             "default-model",
		ContextWindow:     5000,
		Temperature:       0.7,
		NumPredict:        -1,
		TopK:              40,
		TopP:              0.9,
		RepeatPenalty:     1.1,
		Seed:              0,
		LLMTimeoutSeconds: 300,
		Provider:          ProviderLMStudio,
		APIKey:            "",
		MaxIterations:     20,
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("failed to read configuration file: %w", err)
	}

	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse configuration file JSON: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	cfg.LLMEndpoint = normalizeEndpoint(cfg.LLMEndpoint, cfg.Provider)

	return cfg, nil
}

func (c *AgentConfig) Validate() error {
	if c.ContextWindow <= 0 {
		return fmt.Errorf("context_window must be positive")
	}
	if c.Temperature < 0 || c.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}
	if c.LLMTimeoutSeconds <= 0 {
		return fmt.Errorf("llm_timeout_seconds must be positive")
	}
	return nil
}

func normalizeEndpoint(endpoint string, provider ProviderType) string {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return "http://localhost:1234"
	}

	switch provider {
	case ProviderOllama:
		endpoint = strings.TrimSuffix(endpoint, "/")
		return endpoint
	default:
		endpoint = strings.TrimSuffix(endpoint, "/")
		return endpoint
	}
}
