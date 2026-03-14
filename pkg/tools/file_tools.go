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
	"regexp"
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
		Description: "Read the contents of a file at the specified path. Requires argument: 'path' (string). Optional: 'showLines' (boolean) to show line numbers.",
		Parameters: map[string]ToolParameter{
			"path":      {Type: "string", Description: "The absolute or relative path to the file to read."},
			"showLines": {Type: "boolean", Description: "If true, show line numbers (default: false)."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			showLines := false
			if showLinesVal, ok := args["showLines"].(bool); ok {
				showLines = showLinesVal
			}

			fileInfo, err := os.Stat(path)
			if err != nil {
				if os.IsNotExist(err) {
					return "", fmt.Errorf("file not found: '%s'. Use 'list_directory' or 'search_file' to find the correct path", path)
				}
				return "", fmt.Errorf("failed to access file: %w", err)
			}
			if fileInfo.IsDir() {
				return "", fmt.Errorf("'%s' is a directory, not a file. Use 'list_directory' to view its contents", path)
			}

			// Check file size - warn for large files
			if fileInfo.Size() > 500000 {
				return "", fmt.Errorf("file too large (%d bytes). Maximum supported size is 500KB. Use 'run_command' with 'head' or 'tail' to read partial content", fileInfo.Size())
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("failed to read file: %w", err)
			}

			content := string(data)
			if showLines {
				lines := strings.Split(content, "\n")
				var sb strings.Builder
				for i, line := range lines {
					fmt.Fprintf(&sb, "%4d\t%s\n", i+1, line)
				}
				return sb.String(), nil
			}

			return content, nil
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
		Description: "Execute a shell command and return its output. Requires argument: 'command' (string). Uses sh -c for proper shell parsing.",
		Parameters: map[string]ToolParameter{
			"command": {Type: "string", Description: "The command line string to execute in the terminal."},
		},
		Required: []string{"command"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			cmdStr, ok := args["command"].(string)
			if !ok || cmdStr == "" {
				return "", fmt.Errorf("missing or invalid 'command' argument")
			}

			// Security sandbox: block dangerous base commands
			blacklist := []string{"rm", "del", "format", "mkfs", "sudo", "su", "shutdown", "reboot", "dd", "mkfs", ":(){ :|:& };:"}
			lowerCmd := strings.ToLower(cmdStr)
			for _, blocked := range blacklist {
				if strings.Contains(lowerCmd, blocked+" ") || strings.HasPrefix(lowerCmd, blocked) {
					return "", fmt.Errorf("security violation: command '%s' is not allowed in sandbox", blocked)
				}
			}

			// Use sh -c for proper shell parsing (handles quotes, pipes, etc.)
			cmd := exec.CommandContext(ctx, "sh", "-c", cmdStr)

			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr

			err := cmd.Run()
			output := stdout.String()

			if err != nil {
				if stderr.String() != "" {
					return output, fmt.Errorf("command failed: %s", stderr.String())
				}
				return output, fmt.Errorf("command failed: %w", err)
			}

			if output == "" {
				return "(command completed successfully with no output)", nil
			}
			return output, nil
		},
	}
}

// ListDirectoryTool returns a tool that lists directory contents.
func ListDirectoryTool() Tool {
	return Tool{
		Name:        "list_directory",
		Description: "List files and directories at the specified path. Requires argument: 'path' (string). Shows file count summary. Case-insensitive path matching.",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The path of the directory to list files for. Default is current directory. Case-insensitive."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				path = "."
			}

			// Try direct path first
			entries, err := os.ReadDir(path)
			if err != nil {
				// Try case-insensitive search for directory
				pathLower := strings.ToLower(path)
				entries, err = os.ReadDir(".")
				if err != nil {
					return "", fmt.Errorf("failed to list directory: %w", err)
				}

				var foundDir string
				for _, e := range entries {
					if e.IsDir() && strings.ToLower(e.Name()) == pathLower {
						foundDir = e.Name()
						break
					}
				}

				if foundDir != "" {
					entries, err = os.ReadDir(foundDir)
					if err != nil {
						return "", fmt.Errorf("failed to list directory '%s': %w", foundDir, err)
					}
				} else {
					// Try partial match
					var partialDirs []string
					for _, e := range entries {
						if e.IsDir() && strings.Contains(strings.ToLower(e.Name()), pathLower) {
							partialDirs = append(partialDirs, e.Name())
						}
					}
					if len(partialDirs) == 1 {
						entries, err = os.ReadDir(partialDirs[0])
						if err != nil {
							return "", fmt.Errorf("failed to list directory '%s': %w", partialDirs[0], err)
						}
					} else if len(partialDirs) > 1 {
						return "", fmt.Errorf("directory '%s' not found. Did you mean: %s", path, strings.Join(partialDirs, ", "))
					} else {
						return "", fmt.Errorf("directory '%s' not found in current location", path)
					}
				}
			}

			var result strings.Builder
			dirCount := 0
			fileCount := 0

			for _, entry := range entries {
				info, err := entry.Info()
				if err != nil {
					continue
				}

				mode := info.Mode()
				isDir := mode.IsDir()
				size := info.Size()
				timeStr := info.ModTime().Format("Jan 02 15:04")

				if isDir {
					dirCount++
					fmt.Fprintf(&result, "📁 %-30s %10s  %s\n", entry.Name()+"/", "-", timeStr)
				} else {
					fileCount++
					sizeStr := formatSize(size)
					fmt.Fprintf(&result, "📄 %-30s %10s  %s\n", entry.Name(), sizeStr, timeStr)
				}
			}

			// Add summary
			fmt.Fprintf(&result, "\n%d file(s), %d directorie(s)", fileCount, dirCount)

			return result.String(), nil
		},
	}
}

// formatSize formats file size in human readable format
func formatSize(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	}
	if bytes < 1024*1024 {
		return fmt.Sprintf("%.1f KB", float64(bytes)/1024)
	}
	if bytes < 1024*1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	}
	return fmt.Sprintf("%.1f GB", float64(bytes)/(1024*1024*1024))
}

// SearchFileTool returns a tool that searches for a file by name.
func SearchFileTool() Tool {
	return Tool{
		Name:        "search_file",
		Description: "Search for a file by name in the workspace. Matches case-insensitively. Requires argument: 'name' (string).",
		Parameters: map[string]ToolParameter{
			"name": {Type: "string", Description: "The name of the file to search for (case-insensitive match)."},
		},
		Required: []string{"name"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			filename, ok := args["name"].(string)
			if !ok || filename == "" {
				return "", fmt.Errorf("missing or invalid 'name' argument")
			}

			filenameLower := strings.ToLower(filename)
			var matches []string

			// Search in current directory first, then walk
			entries, err := os.ReadDir(".")
			if err == nil {
				for _, entry := range entries {
					if !entry.IsDir() && strings.ToLower(entry.Name()) == filenameLower {
						matches = append(matches, entry.Name())
					}
				}
			}

			// Also search recursively
			err = filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				if d.IsDir() {
					if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" {
						return filepath.SkipDir
					}
					// Check current dir for match already handled above
					return nil
				}
				if strings.ToLower(d.Name()) == filenameLower {
					matches = append(matches, path)
				}
				return nil
			})

			if err != nil {
				return "", fmt.Errorf("search failed: %w", err)
			}

			// Remove duplicates
			seen := make(map[string]bool)
			uniqueMatches := []string{}
			for _, m := range matches {
				if !seen[m] {
					seen[m] = true
					uniqueMatches = append(uniqueMatches, m)
				}
			}

			if len(uniqueMatches) == 0 {
				// Try partial match
				var partialMatches []string
				err = filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
					if err != nil {
						return nil
					}
					if d.IsDir() {
						if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" {
							return filepath.SkipDir
						}
						return nil
					}
					if !d.IsDir() && strings.Contains(strings.ToLower(d.Name()), filenameLower) {
						partialMatches = append(partialMatches, path)
					}
					return nil
				})

				if err == nil && len(partialMatches) > 0 {
					return "Found (partial match):\n" + strings.Join(partialMatches, "\n"), nil
				}

				return "", fmt.Errorf("file '%s' not found in workspace. Try using grep_search to search file contents instead.", filename)
			}

			return "Found at:\n" + strings.Join(uniqueMatches, "\n"), nil
		},
	}
}

// GrepSearchTool returns a tool that searches for a pattern in files.
func GrepSearchTool() Tool {
	return Tool{
		Name:        "grep_search",
		Description: "Search for a string or regex pattern in files. Requires arguments: 'path' (directory to search) and 'query' (string or regex).",
		Parameters: map[string]ToolParameter{
			"path":    {Type: "string", Description: "The directory to search inside."},
			"query":   {Type: "string", Description: "The string or regex pattern to search for."},
			"isRegex": {Type: "boolean", Description: "Set to true if query is a regex pattern (default: false for simple string match)."},
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

			isRegex := false
			if isRegexVal, ok := args["isRegex"].(bool); ok {
				isRegex = isRegexVal
			}

			var result strings.Builder
			count := 0
			maxResults := 100

			err := filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}

				if err != nil {
					return nil
				}
				if d.IsDir() {
					if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" {
						return filepath.SkipDir
					}
					return nil
				}

				// Skip binary files
				ext := strings.ToLower(filepath.Ext(p))
				binaryExts := []string{".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".doc", ".docx"}
				for _, b := range binaryExts {
					if ext == b {
						return nil
					}
				}

				data, err := os.ReadFile(p)
				if err != nil {
					return nil
				}

				content := string(data)
				matched := false

				if isRegex {
					matched, err = matchRegex(query, content)
					if err != nil {
						return nil // Skip invalid regex
					}
				} else {
					matched = strings.Contains(content, query)
				}

				if matched {
					lines := strings.Split(content, "\n")
					for i, line := range lines {
						lineMatch := false
						if isRegex {
							lineMatch, _ = matchRegex(query, line)
						} else {
							lineMatch = strings.Contains(line, query)
						}
						if lineMatch {
							lineNum := i + 1
							trimmedLine := strings.TrimSpace(line)
							if len(trimmedLine) > 150 {
								trimmedLine = trimmedLine[:150] + "..."
							}
							fmt.Fprintf(&result, "%s:%d: %s\n", p, lineNum, trimmedLine)
							count++
							if count >= maxResults {
								fmt.Fprintf(&result, "... (results limited to %d matches) ...\n", maxResults)
								return fmt.Errorf("limit reached")
							}
						}
					}
				}
				return nil
			})

			if err != nil && err.Error() != "limit reached" && err != context.Canceled {
				return "", fmt.Errorf("search failed: %w", err)
			}

			if count == 0 {
				return fmt.Sprintf("No matches found for '%s' in %s", query, path), nil
			}
			return result.String(), nil
		},
	}
}

// matchRegex checks if a pattern matches content
func matchRegex(pattern, content string) (bool, error) {
	// Simple regex matching using strings.Replace for common patterns
	// For full regex, we'd use regexp.Compile
	re, err := regexp.Compile(pattern)
	if err != nil {
		// If regex is invalid, fall back to string match
		return strings.Contains(content, pattern), nil
	}
	return re.MatchString(content), nil
}

// ReplaceFileContentTool allows targeted file replacement.
func ReplaceFileContentTool() Tool {
	return Tool{
		Name:        "replace_file_content",
		Description: "Replaces target string with replacement string in a file. Requires arguments: 'path', 'target', and 'replacement'. Set 'replaceAll' to true to replace all occurrences.",
		Parameters: map[string]ToolParameter{
			"path":        {Type: "string", Description: "The absolute or relative path to the file to modify."},
			"target":      {Type: "string", Description: "The exact string content to find and replace."},
			"replacement": {Type: "string", Description: "The string content to replace it with."},
			"replaceAll":  {Type: "boolean", Description: "If true, replace all occurrences. Default is false (replace first only)."},
		},
		Required: []string{"path", "target", "replacement"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, okPath := args["path"].(string)
			target, okTarget := args["target"].(string)
			replacement, okReplacement := args["replacement"].(string)

			if !okPath || path == "" || !okTarget || !okReplacement {
				return "", fmt.Errorf("missing required 'path', 'target', or 'replacement' argument")
			}

			replaceAll := false
			if replaceAllVal, ok := args["replaceAll"].(bool); ok {
				replaceAll = replaceAllVal
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("failed to read file: %w", err)
			}

			content := string(data)
			if !strings.Contains(content, target) {
				return "", fmt.Errorf("target string not found in file. File content:\n%s", content[:min(len(content), 500)])
			}

			var newContent string
			if replaceAll {
				newContent = strings.ReplaceAll(content, target, replacement)
			} else {
				newContent = strings.Replace(content, target, replacement, 1)
			}

			if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
				return "", fmt.Errorf("failed to write file: %w", err)
			}

			count := strings.Count(newContent, replacement) - strings.Count(content, replacement)
			if count < 0 {
				count = 0
			}
			if replaceAll {
				return fmt.Sprintf("Successfully replaced %d occurrence(s) in %s", count, path), nil
			}
			return fmt.Sprintf("Successfully replaced content in %s", path), nil
		},
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SearchDirectoryTool searches for directories by name (case-insensitive)
func SearchDirectoryTool() Tool {
	return Tool{
		Name:        "search_directory",
		Description: "Search for a directory by name in the workspace. Case-insensitive. Requires argument: 'name' (string).",
		Parameters: map[string]ToolParameter{
			"name": {Type: "string", Description: "The name of the directory to search for (case-insensitive)."},
		},
		Required: []string{"name"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			dirname, ok := args["name"].(string)
			if !ok || dirname == "" {
				return "", fmt.Errorf("missing or invalid 'name' argument")
			}

			dirnameLower := strings.ToLower(dirname)
			var matches []string

			err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				if d.IsDir() {
					if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" || d.Name() == ".vscode" || d.Name() == ".idea" {
						return nil // Skip these directories
					}
					if strings.ToLower(d.Name()) == dirnameLower {
						matches = append(matches, path)
					}
				}
				return nil
			})

			if err != nil {
				return "", fmt.Errorf("search failed: %w", err)
			}

			if len(matches) == 0 {
				// Try partial match
				var partialMatches []string
				err = filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
					if err != nil {
						return nil
					}
					if d.IsDir() {
						if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" {
							return nil
						}
						if strings.Contains(strings.ToLower(d.Name()), dirnameLower) {
							partialMatches = append(partialMatches, path)
						}
					}
					return nil
				})

				if err == nil && len(partialMatches) > 0 {
					return "Found (partial match):\n" + strings.Join(partialMatches, "\n"), nil
				}

				return "", fmt.Errorf("directory '%s' not found in workspace", dirname)
			}

			return "Found at:\n" + strings.Join(matches, "\n"), nil
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

// FileExistsTool checks if a file or directory exists
func FileExistsTool() Tool {
	return Tool{
		Name:        "file_exists",
		Description: "Check if a file or directory exists. Requires argument: 'path' (string).",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The path to check."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			info, err := os.Stat(path)
			if err != nil {
				if os.IsNotExist(err) {
					return "false", nil
				}
				return "", fmt.Errorf("error checking path: %w", err)
			}

			if info.IsDir() {
				return fmt.Sprintf("true (directory: %s)", path), nil
			}
			return fmt.Sprintf("true (file: %s, size: %s)", path, formatSize(info.Size())), nil
		},
	}
}

// GetWorkingDirTool returns the current working directory
func GetWorkingDirTool() Tool {
	return Tool{
		Name:        "get_working_directory",
		Description: "Get the current working directory. No arguments required.",
		Parameters:  map[string]ToolParameter{},
		Required:    []string{},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			dir, err := os.Getwd()
			if err != nil {
				return "", fmt.Errorf("failed to get working directory: %w", err)
			}
			return dir, nil
		},
	}
}

// GetEnvVarsTool returns environment variables
func GetEnvVarsTool() Tool {
	return Tool{
		Name:        "get_env_vars",
		Description: "Get environment variables. Optional argument: 'filter' to filter by prefix.",
		Parameters: map[string]ToolParameter{
			"filter": {Type: "string", Description: "Filter variables by prefix (e.g., 'PATH', 'HOME')."},
		},
		Required: []string{},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			filter, _ := args["filter"].(string)
			filterLower := strings.ToLower(filter)

			var result strings.Builder
			count := 0
			for _, env := range os.Environ() {
				pair := strings.SplitN(env, "=", 2)
				if len(pair) != 2 {
					continue
				}
				if filter != "" && !strings.HasPrefix(strings.ToLower(pair[0]), filterLower) {
					continue
				}
				fmt.Fprintf(&result, "%s=%s\n", pair[0], pair[1])
				count++
				if count >= 50 && filter == "" {
					fmt.Fprintf(&result, "... (showing first 50 of %d variables)", count)
					break
				}
			}
			return result.String(), nil
		},
	}
}

// GlobTool finds files matching a pattern
func GlobTool() Tool {
	return Tool{
		Name:        "glob",
		Description: "Find files matching a pattern. Supports wildcards (* and **). Requires argument: 'pattern' (string).",
		Parameters: map[string]ToolParameter{
			"pattern": {Type: "string", Description: "The glob pattern (e.g., '*.go', '**/*.md')."},
		},
		Required: []string{"pattern"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			pattern, ok := args["pattern"].(string)
			if !ok || pattern == "" {
				return "", fmt.Errorf("missing or invalid 'pattern' argument")
			}

			var matches []string
			err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				if d.IsDir() {
					if d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "vendor" {
						return filepath.SkipDir
					}
					return nil
				}

				name := d.Name()
				// Simple glob matching
				matched := matchGlob(name, pattern)
				if matched {
					matches = append(matches, path)
				}
				return nil
			})

			if err != nil {
				return "", fmt.Errorf("glob failed: %w", err)
			}

			if len(matches) == 0 {
				return "No files match the pattern", nil
			}
			return "Found:\n" + strings.Join(matches, "\n"), nil
		},
	}
}

// matchGlob does simple glob matching
func matchGlob(name, pattern string) bool {
	// Convert glob to regex
	pattern = strings.ReplaceAll(pattern, ".", "\\.")
	pattern = strings.ReplaceAll(pattern, "**/", "((.*/)?)")
	pattern = strings.ReplaceAll(pattern, "**", ".*")
	pattern = strings.ReplaceAll(pattern, "*", "[^/]*")
	pattern = "^" + pattern + "$"

	re, err := regexp.Compile(pattern)
	if err != nil {
		return false
	}
	return re.MatchString(name)
}

// LineCountTool counts lines in a file
func LineCountTool() Tool {
	return Tool{
		Name:        "line_count",
		Description: "Count lines in a file. Requires argument: 'path' (string).",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The file to count lines in."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return "", fmt.Errorf("failed to read file: %w", err)
			}

			lines := strings.Split(string(data), "\n")
			return fmt.Sprintf("%s: %d lines, %d bytes", path, len(lines), len(data)), nil
		},
	}
}

// FileExtensionTool gets file extension
func FileExtensionTool() Tool {
	return Tool{
		Name:        "file_extension",
		Description: "Get file extension. Requires argument: 'path' (string).",
		Parameters: map[string]ToolParameter{
			"path": {Type: "string", Description: "The file path."},
		},
		Required: []string{"path"},
		Execute: func(ctx context.Context, args map[string]interface{}) (string, error) {
			path, ok := args["path"].(string)
			if !ok || path == "" {
				return "", fmt.Errorf("missing or invalid 'path' argument")
			}

			ext := filepath.Ext(path)
			if ext == "" {
				return fmt.Sprintf("No extension (base name: %s)", filepath.Base(path)), nil
			}
			return fmt.Sprintf("Extension: %s (mime type: %s)", ext, getMimeType(ext)), nil
		},
	}
}

func getMimeType(ext string) string {
	mimeTypes := map[string]string{
		".go":         "text/x-go",
		".js":         "application/javascript",
		".ts":         "application/typescript",
		".py":         "text/x-python",
		".java":       "text/x-java",
		".c":          "text/x-c",
		".cpp":        "text/x-c++",
		".h":          "text/x-c-header",
		".rs":         "text/x-rust",
		".rb":         "text/x-ruby",
		".php":        "text/x-php",
		".swift":      "text/x-swift",
		".kt":         "text/x-kotlin",
		".scala":      "text/x-scala",
		".json":       "application/json",
		".xml":        "application/xml",
		".yaml":       "application/x-yaml",
		".yml":        "application/x-yaml",
		".md":         "text/markdown",
		".txt":        "text/plain",
		".html":       "text/html",
		".css":        "text/css",
		".sh":         "application/x-sh",
		".bash":       "application/x-sh",
		".sql":        "text/x-sql",
		".ssh":        "application/x-ssh",
		".env":        "application/x-env",
		".toml":       "application/toml",
		".dockerfile": "text/x-dockerfile",
	}
	return mimeTypes[ext]
}
