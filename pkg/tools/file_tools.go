package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ToolParameter defines the type and description of a parameter
type ToolParameter struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// Tool represents a callable tool with name and execution function.
// Using struct instead of interface{} for type safety (Go vs JS dynamic typing).
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]ToolParameter
	Required    []string
	Execute     func(ctx context.Context, args map[string]interface{}) (string, error)
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
		Description: "Read the contents of a file at the specified path. Requires argument: 'path' (string).",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The absolute or relative path to the file to read."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			fileInfo, err := os.Stat(path)
			if err != nil {
				if os.IsNotExist(err) {
					return "", fmt.Errorf("failed to read file: %w. Tip: The file does not exist. Use 'list_directory' or 'search_file' to find the correct path before trying again", err)
				}
				return "", fmt.Errorf("failed to access file info: %w", err)
			}
			if fileInfo.IsDir() {
				return "", fmt.Errorf("failed to read file: '%s' is a directory, not a file. Use 'list_directory' to view its contents", path)
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
func WriteFileTool() Tool {
	return Tool{
		Name:        "write_file",
		Description: "Write content to a file. Requires arguments: 'path' (string) and 'content' (string).",
		Parameters: map[string]ToolParameter{
			"path":    {Type: "string", Description: "The absolute or relative path to the file to write to."},
			"content": {Type: "string", Description: "The content to write to the file."},
		},
		Required: []string{"path", "content"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			content, ok := args["content"].(string)
			if !ok {
				return "", fmt.Errorf("missing or invalid 'content' argument")
			}

			// Create parent directories if needed
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
func RunCommandTool() Tool {
	return Tool{
		Name:        "run_command",
		Description: "Execute a shell command and return its output. Requires argument: 'command' (string).",
		Parameters: map[string]ToolParameter{
			"command": {Type: "string", Description: "The command line string to execute in the terminal."},
		},
		Required: []string{"command"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			cmdStr, ok := args["command"].(string)
			if !ok || cmdStr == "" {
				return "", fmt.Errorf("missing or invalid 'command' argument")
			}

			// Split command into parts for exec.Command
			parts := strings.Split(cmdStr, " ")
			if len(parts) == 0 {
				return "", fmt.Errorf("empty command")
			}

			// Security sandbox: block dangerous base commands
			blacklist := []string{"rm", "del", "format", "mkfs", "sudo", "su", "shutdown", "reboot"}
			baseCmd := strings.ToLower(parts[0])
			for _, blocked := range blacklist {
				if baseCmd == blocked {
					return "", fmt.Errorf("security violation: execution of command '%s' is not allowed in sandbox", baseCmd)
				}
			}

			cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)

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
func ListDirectoryTool() Tool {
	return Tool{
		Name:        "list_directory",
		Description: "List files and directories at the specified path. Requires argument: 'path' (string).",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The path of the directory to list files for."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				path = "." // Default to current directory if not specified
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

// SearchFileTool returns a tool that searches for a file by name.
func SearchFileTool() Tool {
	return Tool{
		Name:        "search_file",
		Description: "Search for a file by exact name in the workspace. Requires argument: 'name' (string).",
		Parameters: map[string]ToolParameter{
			"name": {Type: "string", Description: "The exact name of the file to search for."},
		},
		Required: []string{"name"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			filename, ok := args["name"].(string)
			if !ok || filename == "" {
				return "", fmt.Errorf("missing or invalid 'name' argument")
			}

			var matches []string
			err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return nil // Skip paths we can't access
				}
				if d.IsDir() && d.Name() == ".git" {
					return filepath.SkipDir
				}
				if !d.IsDir() && d.Name() == filename {
					matches = append(matches, path)
				}
				return nil
			})

			if err != nil {
				return "", fmt.Errorf("search failed: %w", err)
			}

			if len(matches) == 0 {
				return "", fmt.Errorf("file '%s' not found in workspace", filename)
			}

			return "Found at:\n" + strings.Join(matches, "\n"), nil
		},
	}
}

// GrepSearchTool returns a tool that searches for a regex pattern in files.
func GrepSearchTool() Tool {
	return Tool{
		Name:        "grep_search",
		Description: "Search for a regex pattern in files. Requires arguments: 'path' (directory to search) and 'query' (regex string).",
		Parameters: map[string]ToolParameter{
			"path":  {Type: "string", Description: "The directory to search inside."},
			"query": {Type: "string", Description: "The regex string query to search for."},
		},
		Required: []string{"path", "query"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, okPath := args["path"].(string)
			if !okPath || path == "" {
				path = "."
			}
			query, okQuery := args["query"].(string)
			if !okQuery || query == "" {
				return "", fmt.Errorf("missing or invalid 'query' argument")
			}

			var result strings.Builder
			count := 0
			err := filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				if d.IsDir() {
					if d.Name() == ".git" || d.Name() == "node_modules" {
						return filepath.SkipDir
					}
					return nil
				}

				// Read files text to search
				data, err := os.ReadFile(p)
				if err != nil {
					return nil
				}

				content := string(data)
				if strings.Contains(content, query) {
					lines := strings.Split(content, "\n")
					for i, line := range lines {
						if strings.Contains(line, query) {
							fmt.Fprintf(&result, "%s:%d: %s\n", p, i+1, strings.TrimSpace(line))
							count++
							if count >= 100 { // Max results limit
								fmt.Fprintf(&result, "... additional results truncated ...\n")
								return fmt.Errorf("limit reached")
							}
						}
					}
				}
				return nil
			})

			if err != nil && err.Error() != "limit reached" {
				return "", fmt.Errorf("search failed: %w", err)
			}

			if count == 0 {
				return "No matches found.", nil
			}
			return result.String(), nil
		},
	}
}

// ReplaceFileContentTool allows targeted file replacement.
func ReplaceFileContentTool() Tool {
	return Tool{
		Name:        "replace_file_content",
		Description: "Replaces target string with replacement string in a file. Requires arguments: 'path', 'target', and 'replacement'.",
		Parameters: map[string]ToolParameter{
			"path":        {Type: "string", Description: "The absolute or relative path to the file to modify."},
			"target":      {Type: "string", Description: "The exact string content to find and replace."},
			"replacement": {Type: "string", Description: "The string content to replace it with."},
		},
		Required: []string{"path", "target", "replacement"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, okPath := args["path"].(string)
			target, okTarget := args["target"].(string)
			replacement, okReplacement := args["replacement"].(string)

			if !okPath || path == "" || !okTarget || !okReplacement {
				return "", fmt.Errorf("missing required 'path', 'target', or 'replacement' argument")
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("failed to read file: %w", err)
			}

			content := string(data)
			if !strings.Contains(content, target) {
				return "", fmt.Errorf("target string not found in file")
			}

			newContent := strings.Replace(content, target, replacement, 1) // strictly replace first occurence or change to ReplaceAll if needed

			if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
				return "", fmt.Errorf("failed to write file: %w", err)
			}

			return fmt.Sprintf("Successfully replaced content in %s", path), nil
		},
	}
}

// ReadURLTool reads contents from a web page.
func ReadURLTool() Tool {
	return Tool{
		Name:        "read_url",
		Description: "Fetch content from a web URL. Requires argument: 'url' (string).",
		Parameters: map[string]ToolParameter{
			"url": {Type: "string", Description: "The HTTP/HTTPS URL to read."},
		},
		Required: []string{"url"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			url, ok := args["url"].(string)
			if !ok || url == "" {
				return "", fmt.Errorf("missing or invalid 'url' argument")
			}

			req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
			if err != nil {
				return "", fmt.Errorf("invalid request: %w", err)
			}

			// Simple HTTP client setup
			client := &http.Client{Timeout: 10 * time.Second}
			resp, err := client.Do(req)
			if err != nil {
				return "", fmt.Errorf("failed to fetch URL: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return "", fmt.Errorf("HTTP error %d: %s", resp.StatusCode, resp.Status)
			}

			bodyBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return "", fmt.Errorf("failed to read response: %w", err)
			}

			// Optional: In a full system you might run HTML to Markdown parsing here
			// For now, return basic text (trimmed to avoid excessive context usage)
			bodyStr := string(bodyBytes)
			if len(bodyStr) > 4000 {
				bodyStr = bodyStr[:4000] + "\n... (content truncated)"
			}
			return bodyStr, nil
		},
	}
}
