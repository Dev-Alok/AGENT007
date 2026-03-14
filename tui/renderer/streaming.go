package renderer

// Output compliance: This renderer outputs to stdout via fmt.Print
// which is used by TUI callback. This is the allowed exception
// as it's specifically for streaming output formatting.

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
)

var (
	ansiReset   = "\033[0m"
	ansiBold    = "\033[1m"
	ansiItalic  = "\033[3m"
	ansiCyan    = "\033[36m"
	ansiGreen   = "\033[32m"
	ansiYellow  = "\033[33m"
	ansiMagenta = "\033[35m"
	ansiBlue    = "\033[34m"
)

type StreamingRenderer struct {
	mu           sync.Mutex
	buffer       strings.Builder
	lastRendered int
}

func NewStreamingRenderer() *StreamingRenderer {
	return &StreamingRenderer{}
}

// Stream formats and prints markdown content in real-time
func (r *StreamingRenderer) Stream(content string) {
	r.mu.Lock()
	r.buffer.WriteString(content)
	r.mu.Unlock()

	formatted := r.formatPartialMarkdown(content)
	fmt.Print(formatted)
}

// formatPartialMarkdown does basic markdown formatting for streaming
// It handles common patterns that can be formatted incrementally
func (r *StreamingRenderer) formatPartialMarkdown(content string) string {
	result := content

	// Format headers (partial - works even when incomplete)
	headerRegex := regexp.MustCompile(`(?m)^(#{1,6}) (.+)$`)
	result = headerRegex.ReplaceAllStringFunc(result, func(match string) string {
		parts := headerRegex.FindStringSubmatch(match)
		if len(parts) >= 3 {
			level := len(parts[1])
			color := getHeaderColor(level)
			return color + ansiBold + match + ansiReset
		}
		return match
	})

	// Format inline code (works with partial content)
	inlineCodeRegex := regexp.MustCompile("`[^`]+`")
	result = inlineCodeRegex.ReplaceAllString(result, ansiYellow+"$0"+ansiReset)

	// Format bold (needs complete pattern)
	boldRegex := regexp.MustCompile(`\*\*(.+?)\*\*`)
	result = boldRegex.ReplaceAllString(result, ansiBold+"$1"+ansiReset)

	// Format italic (needs complete pattern)
	italicRegex := regexp.MustCompile(`\*(.+?)\*`)
	result = italicRegex.ReplaceAllString(result, ansiItalic+"$1"+ansiReset)

	// Format code blocks (when we see opening fence)
	codeBlockStartRegex := regexp.MustCompile("```.*")
	result = codeBlockStartRegex.ReplaceAllString(result, ansiGreen+"$0"+ansiReset)

	// Format list items
	listRegex := regexp.MustCompile(`(?m)^[-*+] `)
	result = listRegex.ReplaceAllString(result, ansiCyan+"• "+ansiReset)

	// Format numbered lists
	numListRegex := regexp.MustCompile(`(?m)^(\d+)\. `)
	result = numListRegex.ReplaceAllString(result, ansiCyan+"$1. "+ansiReset)

	// Format links (partial - shows URL)
	linkRegex := regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	result = linkRegex.ReplaceAllString(result, ansiBlue+"$1"+ansiReset+"($2)")

	// Format blockquotes
	blockquoteRegex := regexp.MustCompile(`(?m)^> `)
	result = blockquoteRegex.ReplaceAllString(result, ansiMagenta+"| "+ansiReset)

	return result
}

func getHeaderColor(level int) string {
	colors := []string{
		ansiBold + "\033[38;5;225m", // H1 - purple
		ansiBold + "\033[38;5;87m",  // H2 - cyan
		ansiBold + "\033[38;5;228m", // H3 - yellow
		ansiBold + "\033[38;5;223m", // H4 - orange
		ansiBold + "\033[38;5;250m", // H5 - gray
		ansiBold + "\033[38;5;245m", // H6 - dark gray
	}
	if level > 0 && level <= len(colors) {
		return colors[level-1]
	}
	return ansiBold
}

// ClearAndRender clears streamed content and shows properly rendered markdown
func (r *StreamingRenderer) ClearAndRender() string {
	r.mu.Lock()
	content := r.buffer.String()
	r.mu.Unlock()

	if content == "" {
		return ""
	}

	// Use glamour for full rendering
	rendered := RenderMarkdown(content)
	if rendered == content {
		// Glamour didn't render, use our basic formatter
		rendered = r.formatPartialMarkdown(content)
	}

	// Clear the buffer
	r.mu.Lock()
	r.buffer.Reset()
	r.lastRendered = 0
	r.mu.Unlock()

	return rendered
}

// GetBuffer returns current buffer content
func (r *StreamingRenderer) GetBuffer() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.buffer.String()
}

// Reset clears the buffer
func (r *StreamingRenderer) Reset() {
	r.mu.Lock()
	r.buffer.Reset()
	r.lastRendered = 0
	r.mu.Unlock()
}

// UseExternalRenderer tries to use streamdown for rendering if available
func UseExternalRenderer(content string) string {
	// Try streamdown if installed
	cmd := exec.Command("sd", "-w", "80")
	cmd.Stdin = strings.NewReader(content)
	cmd.Stdout = os.Stdout
	cmd.Stderr = nil

	if err := cmd.Run(); err == nil {
		return ""
	}

	// Fall back to basic rendering
	return content
}

// StreamWithPipe pipes content through streamdown if available
func StreamWithPipe(content string) {
	cmd := exec.Command("sd")
	cmd.Stdout = os.Stdout
	cmd.Stderr = nil

	pipe, err := cmd.StdinPipe()
	if err != nil {
		fmt.Print(content)
		return
	}

	fmt.Fprint(pipe, content)
	pipe.Close()
	cmd.Wait()
}

// FormatSimple does basic ANSI formatting without external dependencies
func FormatSimple(content string) string {
	if content == "" {
		return ""
	}

	result := content

	// Headers
	headerRegex := regexp.MustCompile(`(?m)^(#{1,6}) `)
	result = headerRegex.ReplaceAllStringFunc(result, func(match string) string {
		parts := headerRegex.FindStringSubmatch(match)
		if len(parts) >= 3 {
			level := len(parts[1])
			color := getHeaderColor(level)
			return color + match + ansiReset
		}
		return match
	})

	// Inline code
	inlineCodeRegex := regexp.MustCompile("`[^`]+`")
	result = inlineCodeRegex.ReplaceAllString(result, ansiYellow+"\033[7m$0\033[0m")

	// Bold
	boldRegex := regexp.MustCompile(`\*\*(.+?)\*\*`)
	result = boldRegex.ReplaceAllString(result, ansiBold+"$1"+ansiReset)

	// Italic
	italicRegex := regexp.MustCompile(`\*(.+?)\*`)
	result = italicRegex.ReplaceAllString(result, ansiItalic+"$1"+ansiReset)

	// Lists
	listRegex := regexp.MustCompile(`(?m)^[-*+] `)
	result = listRegex.ReplaceAllString(result, ansiCyan+"• "+ansiReset)

	// Numbered lists
	numListRegex := regexp.MustCompile("(?m)^(\\d+)\\. ")
	result = numListRegex.ReplaceAllString(result, ansiCyan+"$1. "+ansiReset)

	return result
}
