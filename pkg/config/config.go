package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// AgentConfig represents the configuration layout for the agent application.
// Used instead of CLI flags to support easy profile switching.
type AgentConfig struct {
	LLMEndpoint       string  `json:"llm_endpoint"`
	Model             string  `json:"model"`
	ContextWindow     int     `json:"context_window"`
	Temperature       float32 `json:"temperature"`
	NumPredict        int     `json:"num_predict"`
	TopK              int     `json:"top_k"`
	TopP              float32 `json:"top_p"`
	RepeatPenalty     float32 `json:"repeat_penalty"`
	Seed              int     `json:"seed"`
	LLMTimeoutSeconds int     `json:"llm_timeout_seconds"`
}

// LoadConfig parses a config.json file and returns an initialized AgentConfig.
// It applies safe defaults so the application can still boot if attributes are missing.
func LoadConfig(filePath string) (*AgentConfig, error) {
	// Provide sane defaults if parsing is fully omitted
	cfg := &AgentConfig{
		LLMEndpoint:       "http://localhost:11434",
		Model:             "llama2",
		ContextWindow:     5000,
		Temperature:       0.7,
		NumPredict:        1000,
		TopK:              40,
		TopP:              0.9,
		RepeatPenalty:     1.1,
		Seed:              0,
		LLMTimeoutSeconds: 30,
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			// If no config file exists, return the defaults
			return cfg, nil
		}
		return nil, fmt.Errorf("failed to read configuration file: %w", err)
	}

	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse configuration file JSON: %w", err)
	}

	return cfg, nil
}
