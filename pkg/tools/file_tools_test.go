package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadFileTool(t *testing.T) {
	// Setup
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test.txt")
	content := "hello world"
	os.WriteFile(filePath, []byte(content), 0644)

	tool := ReadFileTool()

	// Valid test
	args := map[string]interface{}{"path": filePath}
	res, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != content {
		t.Errorf("expected %q, got %q", content, res)
	}

	// Invalid test
	argsInvalid := map[string]interface{}{}
	_, err = tool.Execute(context.Background(), argsInvalid)
	if err == nil {
		t.Error("expected error for missing path")
	}
}

func TestWriteFileTool(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "out.txt")
	content := "new content"

	tool := WriteFileTool()
	args := map[string]interface{}{
		"path":    filePath,
		"content": content,
	}

	_, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}
	if string(data) != content {
		t.Errorf("expected %q, got %q", content, string(data))
	}
}

func TestSearchFileTool(t *testing.T) {
	tmpDir := t.TempDir()
	targetName := "target.go"
	os.WriteFile(filepath.Join(tmpDir, targetName), []byte("test"), 0644)

	// Create subfolder with another file
	subDir := filepath.Join(tmpDir, "sub")
	os.Mkdir(subDir, 0755)
	os.WriteFile(filepath.Join(subDir, "other.go"), []byte("test"), 0644)

	tool := SearchFileTool()

	// Change working dir for the test
	originalDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(originalDir)

	args := map[string]interface{}{"name": targetName}
	res, err := tool.Execute(context.Background(), args)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(res, targetName) {
		t.Errorf("expected result to contain %q, got %q", targetName, res)
	}
}

func TestGrepSearchTool(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "target.go")
	os.WriteFile(filePath, []byte("func main() {\n\tprintln(\"secret\")\n}"), 0644)

	tool := GrepSearchTool()
	args := map[string]interface{}{
		"path":  tmpDir,
		"query": "secret",
	}

	res, err := tool.Execute(context.Background(), args)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(res, "secret") {
		t.Errorf("expected result to contain 'secret', got %q", res)
	}
}

func TestReplaceFileContentTool(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "target.txt")
	os.WriteFile(filePath, []byte("hello old friend"), 0644)

	tool := ReplaceFileContentTool()
	args := map[string]interface{}{
		"path":        filePath,
		"target":      "old",
		"replacement": "new",
	}

	_, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, _ := os.ReadFile(filePath)
	if string(data) != "hello new friend" {
		t.Errorf("expected 'hello new friend', got %q", string(data))
	}
}

func TestReadURLTool(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("mock website content"))
	}))
	defer ts.Close()

	tool := ReadURLTool()
	args := map[string]interface{}{"url": ts.URL}

	res, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != "mock website content" {
		t.Errorf("expected 'mock website content', got %q", res)
	}
}

func TestListDirectoryTool(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "file1.txt"), []byte("data"), 0644)

	tool := ListDirectoryTool()
	args := map[string]interface{}{"path": tmpDir}

	res, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(res, "file1.txt") {
		t.Errorf("expected result to list 'file1.txt', got %q", res)
	}
}

func TestRunCommandTool(t *testing.T) {
	tool := RunCommandTool()

	// Test echo command (windows safe)
	args := map[string]interface{}{"command": "cmd /c echo hello"}

	res, err := tool.Execute(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(res, "hello") {
		t.Errorf("expected result to contain 'hello', got %q", res)
	}
}
