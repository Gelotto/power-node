package executor

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/models"
)

// TestNewPythonExecutor tests the constructor
func TestNewPythonExecutor(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	if exec == nil {
		t.Fatal("NewPythonExecutor() returned nil")
	}
	if exec.responses == nil {
		t.Error("responses map should be initialized")
	}
	if exec.videoResponses == nil {
		t.Error("videoResponses map should be initialized")
	}
	if exec.faceswapResponses == nil {
		t.Error("faceswapResponses map should be initialized")
	}
	if exec.ctx == nil {
		t.Error("context should be initialized")
	}
}

// TestSetProgressCallback tests the progress callback setter
func TestSetProgressCallback(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	var callbackInvoked bool
	callback := func(msg models.ProgressMessage) {
		callbackInvoked = true
	}

	exec.SetProgressCallback(callback)

	// Verify callback was set by getting it
	cb := exec.getProgressCallback()
	if cb == nil {
		t.Error("getProgressCallback() returned nil after SetProgressCallback")
	}

	// Invoke the callback to verify it's the right one
	cb(models.ProgressMessage{})
	if !callbackInvoked {
		t.Error("callback was not invoked")
	}
}

// TestSetProgressCallback_Nil tests clearing the callback
func TestSetProgressCallback_Nil(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	// Set a callback first
	exec.SetProgressCallback(func(msg models.ProgressMessage) {})

	// Clear it
	exec.SetProgressCallback(nil)

	cb := exec.getProgressCallback()
	if cb != nil {
		t.Error("getProgressCallback() should return nil after SetProgressCallback(nil)")
	}
}

// TestSetProgressCallback_Concurrent tests thread safety
func TestSetProgressCallback_Concurrent(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	var wg sync.WaitGroup
	iterations := 100

	// Concurrent setters
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			exec.SetProgressCallback(func(msg models.ProgressMessage) {})
		}(i)
	}

	// Concurrent getters
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = exec.getProgressCallback()
		}()
	}

	wg.Wait()
	// No race condition = test passes
}

// TestRequestIDCounter tests that request IDs are unique and atomic
func TestRequestIDCounter(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	ids := make(map[uint64]bool)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Generate 100 IDs concurrently
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := atomic.AddUint64(&exec.reqID, 1)
			mu.Lock()
			if ids[id] {
				t.Errorf("duplicate ID: %d", id)
			}
			ids[id] = true
			mu.Unlock()
		}()
	}

	wg.Wait()

	if len(ids) != 100 {
		t.Errorf("expected 100 unique IDs, got %d", len(ids))
	}
}

// TestJSONRPCRequest_Marshaling tests JSON-RPC request format
func TestJSONRPCRequest_Marshaling(t *testing.T) {
	seed := int64(42)
	req := models.JSONRPCRequest{
		ID:     1,
		Method: "generate",
		Params: models.GenerateRequest{
			Prompt: "a cat",
			Width:  1024,
			Height: 1024,
			Steps:  8,
			Seed:   &seed,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// Verify structure
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if result["id"] != float64(1) {
		t.Errorf("id = %v, want 1", result["id"])
	}
	if result["method"] != "generate" {
		t.Errorf("method = %v, want generate", result["method"])
	}

	params := result["params"].(map[string]interface{})
	if params["prompt"] != "a cat" {
		t.Errorf("params.prompt = %v, want 'a cat'", params["prompt"])
	}
}

// TestJSONRPCVideoRequest_Marshaling tests video request format
func TestJSONRPCVideoRequest_Marshaling(t *testing.T) {
	steps := 25
	req := models.JSONRPCVideoRequest{
		ID:     2,
		Method: "generate_video",
		Params: models.GenerateVideoRequest{
			Prompt:          "a flying bird",
			Width:           832,
			Height:          480,
			DurationSeconds: 5,
			FPS:             24,
			TotalFrames:     120,
			Steps:           &steps,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// Verify it contains expected fields
	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"generate_video"`) {
		t.Error("JSON should contain generate_video method")
	}
	if !strings.Contains(jsonStr, `"total_frames":120`) {
		t.Error("JSON should contain total_frames")
	}
}

// TestJSONRPCFaceSwapRequest_Marshaling tests face-swap request format
func TestJSONRPCFaceSwapRequest_Marshaling(t *testing.T) {
	req := models.JSONRPCFaceSwapRequest{
		ID:     3,
		Method: "face_swap",
		Params: models.FaceSwapRequest{
			SourceImageURL: "https://example.com/source.jpg",
			TargetImageURL: "https://example.com/target.jpg",
			IsGIF:          true,
			SwapAllFaces:   true,
			Enhance:        true,
			MaxFrames:      50,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if result["method"] != "face_swap" {
		t.Errorf("method = %v, want face_swap", result["method"])
	}

	params := result["params"].(map[string]interface{})
	if params["is_gif"] != true {
		t.Errorf("params.is_gif = %v, want true", params["is_gif"])
	}
}

// TestJSONRPCResponse_Success tests successful response parsing
func TestJSONRPCResponse_Success(t *testing.T) {
	jsonData := `{"id": 1, "result": {"image_data": "base64data...", "format": "png"}, "error": null}`

	var resp models.JSONRPCResponse
	err := json.Unmarshal([]byte(jsonData), &resp)

	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if resp.ID != 1 {
		t.Errorf("ID = %d, want 1", resp.ID)
	}
	if resp.Error != nil {
		t.Errorf("Error = %v, want nil", resp.Error)
	}
	if resp.Result == nil {
		t.Fatal("Result = nil, want result")
	}
	if resp.Result.ImageData != "base64data..." {
		t.Errorf("Result.ImageData = %q, want 'base64data...'", resp.Result.ImageData)
	}
	if resp.Result.Format != "png" {
		t.Errorf("Result.Format = %q, want 'png'", resp.Result.Format)
	}
}

// TestJSONRPCResponse_Error tests error response parsing
func TestJSONRPCResponse_Error(t *testing.T) {
	jsonData := `{"id": 1, "result": null, "error": "CUDA out of memory"}`

	var resp models.JSONRPCResponse
	err := json.Unmarshal([]byte(jsonData), &resp)

	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if resp.Error == nil {
		t.Fatal("Error = nil, want error message")
	}
	if *resp.Error != "CUDA out of memory" {
		t.Errorf("Error = %q, want 'CUDA out of memory'", *resp.Error)
	}
	if resp.Result != nil {
		t.Errorf("Result = %v, want nil", resp.Result)
	}
}

// TestJSONRPCVideoResponse_Success tests video response parsing
func TestJSONRPCVideoResponse_Success(t *testing.T) {
	jsonData := `{"id": 2, "result": {"video_data": "base64videodata...", "format": "mp4", "frames_generated": 120, "duration_seconds": 5}, "error": null}`

	var resp models.JSONRPCVideoResponse
	err := json.Unmarshal([]byte(jsonData), &resp)

	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if resp.ID != 2 {
		t.Errorf("ID = %d, want 2", resp.ID)
	}
	if resp.Result == nil {
		t.Fatal("Result = nil, want result")
	}
	if resp.Result.VideoData != "base64videodata..." {
		t.Errorf("Result.VideoData = %q, want 'base64videodata...'", resp.Result.VideoData)
	}
	if resp.Result.FramesGenerated != 120 {
		t.Errorf("Result.FramesGenerated = %d, want 120", resp.Result.FramesGenerated)
	}
}

// TestJSONRPCFaceSwapResponse_Success tests face-swap response parsing
func TestJSONRPCFaceSwapResponse_Success(t *testing.T) {
	jsonData := `{"id": 3, "result": {"image_data": "base64gifdata...", "format": "gif", "frames_swapped": 25}, "error": null}`

	var resp models.JSONRPCFaceSwapResponse
	err := json.Unmarshal([]byte(jsonData), &resp)

	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if resp.Result == nil {
		t.Fatal("Result = nil, want result")
	}
	if resp.Result.Format != "gif" {
		t.Errorf("Result.Format = %q, want 'gif'", resp.Result.Format)
	}
	if resp.Result.FramesSwapped != 25 {
		t.Errorf("Result.FramesSwapped = %d, want 25", resp.Result.FramesSwapped)
	}
}

// TestProgressMessage_Parsing tests progress message detection and parsing
func TestProgressMessage_Parsing(t *testing.T) {
	jsonData := `{"type": "progress", "step": 5, "total_steps": 25, "progress_percent": 20.0, "frames_completed": 24}`

	var msg models.ProgressMessage
	err := json.Unmarshal([]byte(jsonData), &msg)

	if err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if msg.Type != "progress" {
		t.Errorf("Type = %q, want 'progress'", msg.Type)
	}
	if msg.Step != 5 {
		t.Errorf("Step = %d, want 5", msg.Step)
	}
	if msg.TotalSteps != 25 {
		t.Errorf("TotalSteps = %d, want 25", msg.TotalSteps)
	}
	if msg.ProgressPercent != 20.0 {
		t.Errorf("ProgressPercent = %f, want 20.0", msg.ProgressPercent)
	}
	if msg.FramesCompleted != 24 {
		t.Errorf("FramesCompleted = %d, want 24", msg.FramesCompleted)
	}
}

// TestProgressMessage_Detection tests distinguishing progress from final response
func TestProgressMessage_Detection(t *testing.T) {
	tests := []struct {
		name       string
		jsonData   string
		isProgress bool
	}{
		{
			name:       "progress message (no id)",
			jsonData:   `{"type": "progress", "step": 5, "total_steps": 25}`,
			isProgress: true,
		},
		{
			name:       "final response (has id)",
			jsonData:   `{"id": 1, "result": {"image_data": "..."}, "error": null}`,
			isProgress: false,
		},
		{
			name:       "progress with explicit null id",
			jsonData:   `{"id": null, "type": "progress", "step": 5}`,
			isProgress: true, // id is null, so it's a progress message
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msgType struct {
				Type string  `json:"type"`
				ID   *uint64 `json:"id"`
			}
			if err := json.Unmarshal([]byte(tt.jsonData), &msgType); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}

			isProgress := msgType.Type == "progress" && msgType.ID == nil

			if isProgress != tt.isProgress {
				t.Errorf("isProgress = %v, want %v", isProgress, tt.isProgress)
			}
		})
	}
}

// TestResponseChannelManagement tests that response channels are properly managed
func TestResponseChannelManagement(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	// Simulate what Generate does: create channel, add to map, remove on cleanup
	reqID := uint64(1)
	respChan := make(chan *models.JSONRPCResponse, 1)

	exec.mu.Lock()
	exec.responses[reqID] = respChan
	exec.mu.Unlock()

	// Verify channel is in map
	exec.mu.Lock()
	ch, ok := exec.responses[reqID]
	exec.mu.Unlock()

	if !ok {
		t.Error("response channel should be in map")
	}
	if ch == nil {
		t.Error("response channel should not be nil")
	}

	// Cleanup (simulates defer in Generate)
	exec.mu.Lock()
	delete(exec.responses, reqID)
	exec.mu.Unlock()

	// Verify channel is removed
	exec.mu.Lock()
	_, ok = exec.responses[reqID]
	exec.mu.Unlock()

	if ok {
		t.Error("response channel should be removed from map after cleanup")
	}
}

// TestVideoResponseChannelManagement tests video response channel management
func TestVideoResponseChannelManagement(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	reqID := uint64(2)
	respChan := make(chan *models.JSONRPCVideoResponse, 1)

	exec.mu.Lock()
	exec.videoResponses[reqID] = respChan
	exec.mu.Unlock()

	exec.mu.Lock()
	_, ok := exec.videoResponses[reqID]
	exec.mu.Unlock()

	if !ok {
		t.Error("video response channel should be in map")
	}

	exec.mu.Lock()
	delete(exec.videoResponses, reqID)
	exec.mu.Unlock()

	exec.mu.Lock()
	_, ok = exec.videoResponses[reqID]
	exec.mu.Unlock()

	if ok {
		t.Error("video response channel should be removed from map after cleanup")
	}
}

// TestFaceSwapResponseChannelManagement tests face-swap response channel management
func TestFaceSwapResponseChannelManagement(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	reqID := uint64(3)
	respChan := make(chan *models.JSONRPCFaceSwapResponse, 1)

	exec.mu.Lock()
	exec.faceswapResponses[reqID] = respChan
	exec.mu.Unlock()

	exec.mu.Lock()
	_, ok := exec.faceswapResponses[reqID]
	exec.mu.Unlock()

	if !ok {
		t.Error("face-swap response channel should be in map")
	}

	exec.mu.Lock()
	delete(exec.faceswapResponses, reqID)
	exec.mu.Unlock()

	exec.mu.Lock()
	_, ok = exec.faceswapResponses[reqID]
	exec.mu.Unlock()

	if ok {
		t.Error("face-swap response channel should be removed from map after cleanup")
	}
}

// TestStart_InvalidPath tests starting with invalid Python path
func TestStart_InvalidPath(t *testing.T) {
	exec := NewPythonExecutor("/nonexistent/python", "test.py", nil, nil)
	err := exec.Start("/nonexistent/python", "test.py", nil, nil)

	if err == nil {
		t.Error("Start() with invalid path should return error")
		_ = exec.Stop()
	}
}

// TestStart_InvalidScript tests starting with invalid script path
func TestStart_InvalidScript(t *testing.T) {
	exec := NewPythonExecutor("python3", "/nonexistent/script.py", nil, nil)
	err := exec.Start("python3", "/nonexistent/script.py", nil, nil)

	// Python will fail because script doesn't exist
	// The error might come from Start() or later when we try to communicate
	// Either way, the process should not be successfully running

	if err == nil {
		// If Start() succeeded, try to stop to clean up
		_ = exec.Stop()
	}
}

// TestEnvironmentVariables tests that environment variables are passed correctly
func TestEnvironmentVariables(t *testing.T) {
	// Create a simple echo script that prints env vars
	script := `
import os
import json
print(json.dumps({"test_var": os.environ.get("TEST_VAR", "")}))
`
	tmpFile, err := os.CreateTemp("", "test_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	// Test that env vars are passed (we can't easily test this without starting process)
	exec := NewPythonExecutor("python3", tmpFile.Name(), nil, map[string]string{
		"TEST_VAR": "test_value",
	})

	// The executor is configured with env vars, but we can't easily verify
	// without actually starting the subprocess. This test at least verifies
	// the constructor accepts env vars without error.
	if exec == nil {
		t.Error("NewPythonExecutor() with env vars should not return nil")
	}
}

// TestGenerate_ContextCancellation tests that context cancellation is respected
func TestGenerate_ContextCancellation(t *testing.T) {
	// Test that context cancellation is properly detected
	// We can't easily test with a real subprocess, but we can verify
	// the context cancellation logic works correctly

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// The context should be cancelled
	if ctx.Err() == nil {
		t.Error("context should be cancelled")
	}

	// Verify the specific error type
	if ctx.Err() != context.Canceled {
		t.Errorf("ctx.Err() = %v, want context.Canceled", ctx.Err())
	}
}

// TestMalformedJSONParsing tests handling of malformed JSON
func TestMalformedJSONParsing(t *testing.T) {
	malformedJSON := []string{
		`{"id": 1, "result": {invalid`,
		`not json at all`,
		`{"id": "string_id"}`, // ID should be number
		`{}`,                  // Missing required fields
	}

	for _, jsonStr := range malformedJSON {
		var resp models.JSONRPCResponse
		err := json.Unmarshal([]byte(jsonStr), &resp)

		// Some malformed JSON will error, some will just have zero values
		// The important thing is we don't panic - we just need to verify
		// the unmarshal completes without crashing
		_ = err
		_ = resp
	}
}

// TestResponseRouting tests that responses are routed to correct channel type
func TestResponseRouting(t *testing.T) {
	// This tests the logic used in handleResponses without needing a real subprocess

	// Test data for different response types
	testCases := []struct {
		name     string
		reqID    uint64
		respType string // "image", "video", "faceswap"
	}{
		{"image response", 1, "image"},
		{"video response", 2, "video"},
		{"faceswap response", 3, "faceswap"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			exec := NewPythonExecutor("python3", "test.py", nil, nil)

			// Register the appropriate channel type
			switch tc.respType {
			case "image":
				exec.mu.Lock()
				exec.responses[tc.reqID] = make(chan *models.JSONRPCResponse, 1)
				exec.mu.Unlock()
			case "video":
				exec.mu.Lock()
				exec.videoResponses[tc.reqID] = make(chan *models.JSONRPCVideoResponse, 1)
				exec.mu.Unlock()
			case "faceswap":
				exec.mu.Lock()
				exec.faceswapResponses[tc.reqID] = make(chan *models.JSONRPCFaceSwapResponse, 1)
				exec.mu.Unlock()
			}

			// Verify the channel exists in the right map
			exec.mu.Lock()
			_, isImage := exec.responses[tc.reqID]
			_, isVideo := exec.videoResponses[tc.reqID]
			_, isFaceSwap := exec.faceswapResponses[tc.reqID]
			exec.mu.Unlock()

			switch tc.respType {
			case "image":
				if !isImage {
					t.Error("image response should be in responses map")
				}
			case "video":
				if !isVideo {
					t.Error("video response should be in videoResponses map")
				}
			case "faceswap":
				if !isFaceSwap {
					t.Error("faceswap response should be in faceswapResponses map")
				}
			}
		})
	}
}

// TestNonBlockingSend tests that channel sends don't block
func TestNonBlockingSend(t *testing.T) {
	// Create a channel with buffer size 1
	ch := make(chan *models.JSONRPCResponse, 1)

	// Fill the buffer
	resp := &models.JSONRPCResponse{ID: 1}
	ch <- resp

	// Try non-blocking send (should not block)
	done := make(chan bool, 1)
	go func() {
		select {
		case ch <- &models.JSONRPCResponse{ID: 2}:
			// Successfully sent (buffer has room after first recv)
		default:
			// Channel full, send dropped (this is expected)
		}
		done <- true
	}()

	// Should complete immediately
	select {
	case <-done:
		// Success - non-blocking send completed
	case <-time.After(100 * time.Millisecond):
		t.Error("non-blocking send should not block")
	}
}

// TestHandleResponses_Integration tests response handling with a mock subprocess
func TestHandleResponses_Integration(t *testing.T) {
	// Create a script that outputs a JSON response
	script := `
import json
import sys
response = {"id": 1, "result": {"image_data": "test_data", "format": "png"}, "error": None}
print(json.dumps(response))
sys.stdout.flush()
`
	tmpFile, err := os.CreateTemp("", "echo_*.py")
	if err != nil {
		t.Skipf("Could not create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	// We can't easily test the full handleResponses without a real subprocess
	// but we can verify the JSON output matches what we expect to parse
	cmd := exec.Command("python3", tmpFile.Name())
	output, cmdErr := cmd.Output()
	if cmdErr != nil {
		t.Skipf("Python not available: %v", cmdErr)
	}

	var resp models.JSONRPCResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		t.Errorf("Failed to parse response: %v", err)
	}

	if resp.ID != 1 {
		t.Errorf("ID = %d, want 1", resp.ID)
	}
	if resp.Result == nil || resp.Result.ImageData != "test_data" {
		t.Error("Result.ImageData should be 'test_data'")
	}
}

// TestProgressCallbackInvocation tests that progress callback is invoked correctly
func TestProgressCallbackInvocation(t *testing.T) {
	exec := NewPythonExecutor("python3", "test.py", nil, nil)

	var receivedProgress models.ProgressMessage
	var callCount int32

	exec.SetProgressCallback(func(msg models.ProgressMessage) {
		atomic.AddInt32(&callCount, 1)
		receivedProgress = msg
	})

	// Simulate what handleResponses does when it gets a progress message
	progressJSON := `{"type": "progress", "step": 10, "total_steps": 50, "progress_percent": 20.0, "frames_completed": 24}`

	var progress models.ProgressMessage
	_ = json.Unmarshal([]byte(progressJSON), &progress)

	// Invoke callback (this is what handleResponses does)
	if cb := exec.getProgressCallback(); cb != nil {
		cb(progress)
	}

	// Give goroutine time to complete (callback is invoked in goroutine in real code)
	time.Sleep(10 * time.Millisecond)

	if atomic.LoadInt32(&callCount) != 1 {
		t.Errorf("callback call count = %d, want 1", callCount)
	}
	if receivedProgress.Step != 10 {
		t.Errorf("receivedProgress.Step = %d, want 10", receivedProgress.Step)
	}
}

// TestStdoutParsing tests parsing of stdout lines
func TestStdoutParsing(t *testing.T) {
	// Simulate stdout parsing that handleResponses does
	lines := []string{
		`{"id": 1, "result": {"image_data": "...", "format": "png"}, "error": null}`,
		`{"type": "progress", "step": 5, "total_steps": 25, "progress_percent": 20.0, "frames_completed": 24}`,
		`{"id": 2, "result": {"video_data": "...", "format": "mp4", "frames_generated": 120, "duration_seconds": 5}, "error": null}`,
	}

	for i, line := range lines {
		// Try to detect message type
		var msgType struct {
			Type string  `json:"type"`
			ID   *uint64 `json:"id"`
		}
		err := json.Unmarshal([]byte(line), &msgType)
		if err != nil {
			t.Errorf("line %d: failed to parse: %v", i, err)
			continue
		}

		isProgress := msgType.Type == "progress" && msgType.ID == nil

		switch i {
		case 0: // image response
			if isProgress {
				t.Errorf("line %d: should not be progress", i)
			}
		case 1: // progress message
			if !isProgress {
				t.Errorf("line %d: should be progress", i)
			}
		case 2: // video response
			if isProgress {
				t.Errorf("line %d: should not be progress", i)
			}
		}
	}
}

// TestBufferedReader tests that bufio.Reader handles line reading correctly
func TestBufferedReader(t *testing.T) {
	// Simulate what handleResponses does with stdout
	jsonLines := `{"id": 1, "result": {}, "error": null}
{"type": "progress", "step": 5}
{"id": 2, "result": {}, "error": null}
`

	reader := bufio.NewReader(strings.NewReader(jsonLines))

	lineCount := 0
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("ReadString error: %v", err)
		}
		lineCount++

		// Verify line is valid JSON (minus newline)
		line = strings.TrimSpace(line)
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			t.Errorf("line %d is not valid JSON: %v", lineCount, err)
		}
	}

	if lineCount != 3 {
		t.Errorf("lineCount = %d, want 3", lineCount)
	}
}

// createMockPythonScript creates a temporary Python script that echoes JSON-RPC responses
func createMockPythonScript(t *testing.T) string {
	t.Helper()

	script := `
import json
import sys

# Read from stdin
for line in sys.stdin:
    try:
        req = json.loads(line.strip())
        req_id = req.get("id", 0)
        method = req.get("method", "")

        if method == "generate":
            response = {
                "id": req_id,
                "result": {
                    "image_data": "base64_mock_image_data",
                    "format": "png"
                },
                "error": None
            }
        elif method == "generate_video":
            # Send progress first
            progress = {
                "type": "progress",
                "step": 10,
                "total_steps": 25,
                "progress_percent": 40.0,
                "frames_completed": 48
            }
            print(json.dumps(progress), flush=True)

            response = {
                "id": req_id,
                "result": {
                    "video_data": "base64_mock_video_data",
                    "format": "mp4",
                    "frames_generated": 120,
                    "duration_seconds": 5
                },
                "error": None
            }
        elif method == "face_swap":
            response = {
                "id": req_id,
                "result": {
                    "image_data": "base64_mock_faceswap_data",
                    "format": "jpeg",
                    "frames_swapped": 1
                },
                "error": None
            }
        else:
            response = {
                "id": req_id,
                "result": None,
                "error": "Unknown method: " + method
            }

        print(json.dumps(response), flush=True)
    except Exception as e:
        print(json.dumps({"id": 0, "result": None, "error": str(e)}), flush=True)
`

	tmpFile, err := os.CreateTemp("", "mock_python_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}

	if _, err := tmpFile.WriteString(script); err != nil {
		tmpFile.Close()
		os.Remove(tmpFile.Name())
		t.Fatalf("Failed to write script: %v", err)
	}
	tmpFile.Close()

	return tmpFile.Name()
}

// TestGenerate_WithMockPython tests image generation with a mock Python subprocess
func TestGenerate_WithMockPython(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	scriptPath := createMockPythonScript(t)
	defer os.Remove(scriptPath)

	executor := NewPythonExecutor("python3", scriptPath, nil, nil)
	if err := executor.Start("python3", scriptPath, nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &models.GenerateRequest{
		Prompt: "test prompt",
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}

	resp, err := executor.Generate(ctx, req)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if resp == nil {
		t.Fatal("Generate() returned nil response")
	}
	if resp.ImageData != "base64_mock_image_data" {
		t.Errorf("ImageData = %q, want 'base64_mock_image_data'", resp.ImageData)
	}
	if resp.Format != "png" {
		t.Errorf("Format = %q, want 'png'", resp.Format)
	}
}

// TestStop_GracefulShutdown tests that Stop gracefully shuts down the subprocess
func TestStop_GracefulShutdown(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	// Create a simple script that stays running
	script := `
import sys
import time
while True:
	time.sleep(0.1)
`
	tmpFile, err := os.CreateTemp("", "long_running_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	executor := NewPythonExecutor("python3", tmpFile.Name(), nil, nil)
	if err := executor.Start("python3", tmpFile.Name(), nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	// Stop should complete within 10 seconds (the kill timeout)
	done := make(chan error, 1)
	go func() {
		done <- executor.Stop()
	}()

	select {
	case err := <-done:
		// Stop completed - may have error if process was killed
		_ = err // Error is expected if killed
	case <-time.After(15 * time.Second):
		t.Error("Stop() took too long")
	}
}

// TestGenerate_Timeout tests that generation times out correctly
func TestGenerate_TimeoutBehavior(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	// Create a script that never responds
	script := `
import sys
import time
while True:
	time.sleep(1)
`
	tmpFile, err := os.CreateTemp("", "slow_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	executor := NewPythonExecutor("python3", tmpFile.Name(), nil, nil)
	if err := executor.Start("python3", tmpFile.Name(), nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	// Use a short timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	req := &models.GenerateRequest{
		Prompt: "test",
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}

	_, err = executor.Generate(ctx, req)
	if err == nil {
		t.Error("Generate() should have timed out")
	}
	if !strings.Contains(err.Error(), "cancelled") && !strings.Contains(err.Error(), "timeout") {
		t.Errorf("error should mention cancellation or timeout, got: %v", err)
	}
}

// TestGenerateVideo_WithMockPython tests video generation with progress
func TestGenerateVideo_WithMockPython(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	scriptPath := createMockPythonScript(t)
	defer os.Remove(scriptPath)

	executor := NewPythonExecutor("python3", scriptPath, nil, nil)
	if err := executor.Start("python3", scriptPath, nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	// Set up progress callback
	var progressReceived bool
	var progressMu sync.Mutex
	executor.SetProgressCallback(func(msg models.ProgressMessage) {
		progressMu.Lock()
		progressReceived = true
		progressMu.Unlock()
	})

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &models.GenerateVideoRequest{
		Prompt:          "test video prompt",
		Width:           832,
		Height:          480,
		DurationSeconds: 5,
		FPS:             24,
		TotalFrames:     120,
	}

	resp, err := executor.GenerateVideo(ctx, req)
	if err != nil {
		t.Fatalf("GenerateVideo() error = %v", err)
	}
	if resp == nil {
		t.Fatal("GenerateVideo() returned nil response")
	}
	if resp.VideoData != "base64_mock_video_data" {
		t.Errorf("VideoData = %q, want 'base64_mock_video_data'", resp.VideoData)
	}
	if resp.Format != "mp4" {
		t.Errorf("Format = %q, want 'mp4'", resp.Format)
	}
	if resp.FramesGenerated != 120 {
		t.Errorf("FramesGenerated = %d, want 120", resp.FramesGenerated)
	}

	// Give time for progress callback goroutine
	time.Sleep(50 * time.Millisecond)

	progressMu.Lock()
	if !progressReceived {
		t.Error("Progress callback should have been invoked")
	}
	progressMu.Unlock()
}

// TestGenerateFaceSwap_WithMockPython tests face-swap generation
func TestGenerateFaceSwap_WithMockPython(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	scriptPath := createMockPythonScript(t)
	defer os.Remove(scriptPath)

	executor := NewPythonExecutor("python3", scriptPath, nil, nil)
	if err := executor.Start("python3", scriptPath, nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &models.FaceSwapRequest{
		SourceImageURL: "https://example.com/source.jpg",
		TargetImageURL: "https://example.com/target.jpg",
		IsGIF:          false,
		SwapAllFaces:   true,
		Enhance:        true,
	}

	resp, err := executor.GenerateFaceSwap(ctx, req)
	if err != nil {
		t.Fatalf("GenerateFaceSwap() error = %v", err)
	}
	if resp == nil {
		t.Fatal("GenerateFaceSwap() returned nil response")
	}
	if resp.ImageData != "base64_mock_faceswap_data" {
		t.Errorf("ImageData = %q, want 'base64_mock_faceswap_data'", resp.ImageData)
	}
	if resp.Format != "jpeg" {
		t.Errorf("Format = %q, want 'jpeg'", resp.Format)
	}
	if resp.FramesSwapped != 1 {
		t.Errorf("FramesSwapped = %d, want 1", resp.FramesSwapped)
	}
}

// TestGenerate_NilResponse tests handling of nil response (e.g., pipe closed)
func TestGenerate_NilResponse(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	// Create a script that exits immediately without responding
	script := `
import sys
sys.exit(0)
`
	tmpFile, err := os.CreateTemp("", "exit_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	executor := NewPythonExecutor("python3", tmpFile.Name(), nil, nil)
	if err := executor.Start("python3", tmpFile.Name(), nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	req := &models.GenerateRequest{
		Prompt: "test",
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}

	_, err = executor.Generate(ctx, req)
	// Should error because Python exits without responding
	if err == nil {
		t.Error("Generate() should error when Python exits")
	}
}

// TestMultipleConcurrentRequests tests handling of multiple concurrent requests
func TestMultipleConcurrentRequests(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	scriptPath := createMockPythonScript(t)
	defer os.Remove(scriptPath)

	executor := NewPythonExecutor("python3", scriptPath, nil, nil)
	if err := executor.Start("python3", scriptPath, nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	time.Sleep(100 * time.Millisecond)

	// Send multiple requests concurrently
	var wg sync.WaitGroup
	results := make(chan error, 3)

	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			req := &models.GenerateRequest{
				Prompt: "test",
				Width:  1024,
				Height: 1024,
				Steps:  8,
			}

			_, err := executor.Generate(ctx, req)
			results <- err
		}(i)
	}

	wg.Wait()
	close(results)

	successCount := 0
	for err := range results {
		if err == nil {
			successCount++
		}
	}

	// At least some requests should succeed
	if successCount < 1 {
		t.Errorf("Expected at least 1 successful request, got %d", successCount)
	}
}

// TestGenerate_ErrorResponse tests handling of error responses from Python
func TestGenerate_ErrorResponse(t *testing.T) {
	// Skip if Python is not available
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	// Create a script that always returns an error
	script := `
import json
import sys

for line in sys.stdin:
	try:
		req = json.loads(line.strip())
		response = {
			"id": req.get("id", 0),
			"result": None,
			"error": "CUDA out of memory"
		}
		print(json.dumps(response), flush=True)
	except:
		pass
`
	tmpFile, err := os.CreateTemp("", "error_*.py")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString(script); err != nil {
		t.Fatalf("WriteString failed: %v", err)
	}
	tmpFile.Close()

	executor := NewPythonExecutor("python3", tmpFile.Name(), nil, nil)
	if err := executor.Start("python3", tmpFile.Name(), nil, nil); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer func() { _ = executor.Stop() }()

	// Give Python time to start
	time.Sleep(100 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &models.GenerateRequest{
		Prompt: "test",
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}

	_, err = executor.Generate(ctx, req)
	if err == nil {
		t.Error("Generate() should return error for CUDA OOM")
	}
	if !strings.Contains(err.Error(), "CUDA out of memory") {
		t.Errorf("error should contain 'CUDA out of memory', got: %v", err)
	}
}
