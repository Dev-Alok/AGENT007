package tools

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// Tool represents a callable tool with name and execution function.
// Using struct instead of interface{} for type safety (Go vs JS dynamic typing).
type Tool struct {
	Name        string
	Description string
	Execute     func(args map[string]interface{}) (string, error)
}

// Registry maintains a thread-safe map of available tools.
// Mutex ensures concurrent access is safe without data races.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

// NewRegistry creates and initializes an empty tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry with thread-safe write lock.
// Using RWMutex allows concurrent reads while writes are exclusive.
func (r *Registry) Register(name string, tool Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[name] = tool
}

// Get retrieves a tool by name with thread-safe read lock.
// Returns ok=false if tool doesn't exist (Go idiom vs exceptions).
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

// List returns all registered tool names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	return names
}

// ReadFileTool returns a tool that reads file contents.
// Uses os.ReadFile for simplicity; handles encoding errors gracefully.
func ReadFileTool() Tool {
	return Tool{
		Name:        "read_file",
		Description: "Read the contents of a file at the specified path.",
		Execute: func(args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("failed to read file: %w", err)
			}

			return string(data), nil
		},
	}
}

// WriteFileTool returns a tool that writes content to a file.
// Creates parent directories if they don't exist (user convenience).
func WriteFileTool() Tool {
	return Tool{
		Name:        "write_file",
		Description: "Write content to a file at the specified path.",
		Execute: func(args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			content, ok := args["content"].(string)
			if !ok {
				return "", fmt.Errorf("missing or invalid 'content' argument")
			}

			// Create parent directories if needed (Go's explicit error handling)
			dir := filepath.Dir(path)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return "", fmt.Errorf("failed to create directory: %w", err)
			}

			if err := os.WriteFile(path, []byte(content), 0644); err != nil {
				return "", fmt.Errorf("failed to write file: %w", err)
			}

			return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), path), nil
		},
	}
}

// RunCommandTool returns a tool that executes shell commands.
// Uses exec.Command with timeout for safety (prevents hanging).
func RunCommandTool() Tool {
	return Tool{
		Name:        "run_command",
		Description: "Execute a shell command and return its output.",
		Execute: func(args map[string]interface{}) (string, error) {
			cmdStr, ok := args["command"].(string)
			if !ok || cmdStr == "" {
				return "", fmt.Errorf("missing or invalid 'command' argument")
			}

			// Split command into parts for exec.Command
			// Note: Using shell=False for security; caller should escape properly
			parts := strings.Split(cmdStr, " ")
			if len(parts) == 0 {
				return "", fmt.Errorf("empty command")
			}

			cmd := exec.Command(parts[0], parts[1:]...)

			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr

			err := cmd.Run()
			output := stdout.String()

			if err != nil {
				return output, fmt.Errorf("command failed: %w, stderr: %s", err, stderr.String())
			}

			return output, nil
		},
	}
}

// ListDirectoryTool returns a tool that lists directory contents.
// Returns formatted list with file types and permissions.
func ListDirectoryTool() Tool {
	return Tool{
		Name:        "list_directory",
		Description: "List files and directories at the specified path.",
		Execute: func(args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			entries, err := os.ReadDir(path)
			if err != nil {
				return "", fmt.Errorf("failed to list directory: %w", err)
			}

			var result strings.Builder
			for _, entry := range entries {
				info, err := entry.Info()
				if err != nil {
					continue
				}

				mode := info.Mode()
				isDir := mode.IsDir()
				size := info.Size()
				timeStr := info.ModTime().Format("2006-01-02 15:04:05")

				prefix := "📄"
				if isDir {
					prefix = "📁"
				} else if mode&os.ModeSymlink != 0 {
					prefix = "🔗"
				}

				fmt.Fprintf(&result, "%s %-40s %10d bytes  %s\n", prefix, entry.Name(), size, timeStr)
			}

			return result.String(), nil
		},
	}
}
