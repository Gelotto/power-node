package executor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Gelotto/power-node/internal/models"
)

// ProgressCallback is called when a progress message is received during video generation
type ProgressCallback func(msg models.ProgressMessage)

// PythonExecutor manages the Python subprocess for image and video generation
type PythonExecutor struct {
	cmd              *exec.Cmd
	stdin            io.WriteCloser
	stdout           *bufio.Reader
	stderr           io.ReadCloser
	reqID            uint64
	responses        map[uint64]chan *models.JSONRPCResponse
	videoResponses   map[uint64]chan *models.JSONRPCVideoResponse
	mu               sync.Mutex
	ctx              context.Context
	cancel           context.CancelFunc
	progressCallback ProgressCallback // Callback for video progress updates
	progressMu       sync.RWMutex     // Separate mutex for callback to avoid deadlock
}

// NewPythonExecutor creates a new Python executor
func NewPythonExecutor(pythonExec, scriptPath string, scriptArgs []string, env map[string]string) *PythonExecutor {
	ctx, cancel := context.WithCancel(context.Background())

	return &PythonExecutor{
		responses:      make(map[uint64]chan *models.JSONRPCResponse),
		videoResponses: make(map[uint64]chan *models.JSONRPCVideoResponse),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// Start launches the Python process
func (e *PythonExecutor) Start(pythonExec, scriptPath string, scriptArgs []string, env map[string]string) error {
	args := append([]string{scriptPath}, scriptArgs...)
	e.cmd = exec.CommandContext(e.ctx, pythonExec, args...)

	e.cmd.Env = os.Environ()
	e.cmd.Env = append(e.cmd.Env, "PYTHONUNBUFFERED=1")
	for k, v := range env {
		e.cmd.Env = append(e.cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	var err error

	e.stdin, err = e.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdoutPipe, err := e.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	e.stdout = bufio.NewReader(stdoutPipe)

	e.stderr, err = e.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	if err := e.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start Python process: %w", err)
	}

	log.Printf("Python process started (PID: %d)", e.cmd.Process.Pid)

	go e.handleResponses()
	go e.logStderr()

	return nil
}

// Generate sends a generation request to Python and waits for response
func (e *PythonExecutor) Generate(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
	reqID := atomic.AddUint64(&e.reqID, 1)

	respChan := make(chan *models.JSONRPCResponse, 1)
	e.mu.Lock()
	e.responses[reqID] = respChan
	e.mu.Unlock()

	defer func() {
		e.mu.Lock()
		delete(e.responses, reqID)
		e.mu.Unlock()
		close(respChan)
	}()

	jsonReq := models.JSONRPCRequest{
		ID:     reqID,
		Method: "generate",
		Params: *req,
	}

	reqData, err := json.Marshal(jsonReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	log.Printf("Sending to Python: %s", string(reqData))

	if _, err := e.stdin.Write(append(reqData, '\n')); err != nil {
		return nil, fmt.Errorf("failed to write to Python stdin: %w", err)
	}

	select {
	case resp := <-respChan:
		if resp == nil {
			return nil, fmt.Errorf("received nil response")
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("Python error: %s", *resp.Error)
		}
		if resp.Result == nil {
			return nil, fmt.Errorf("received empty result")
		}
		return resp.Result, nil

	case <-ctx.Done():
		return nil, fmt.Errorf("request cancelled: %w", ctx.Err())

	case <-time.After(300 * time.Second):
		return nil, fmt.Errorf("generation timeout (5 minutes)")
	}
}

// GenerateVideo sends a video generation request to Python and waits for response
// Video generation has a longer timeout (30 minutes) due to the complexity of video generation
func (e *PythonExecutor) GenerateVideo(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error) {
	reqID := atomic.AddUint64(&e.reqID, 1)

	respChan := make(chan *models.JSONRPCVideoResponse, 1)
	e.mu.Lock()
	e.videoResponses[reqID] = respChan
	e.mu.Unlock()

	defer func() {
		e.mu.Lock()
		delete(e.videoResponses, reqID)
		e.mu.Unlock()
		close(respChan)
	}()

	jsonReq := models.JSONRPCVideoRequest{
		ID:     reqID,
		Method: "generate_video",
		Params: *req,
	}

	reqData, err := json.Marshal(jsonReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal video request: %w", err)
	}

	log.Printf("Sending video request to Python: %s", string(reqData))

	if _, err := e.stdin.Write(append(reqData, '\n')); err != nil {
		return nil, fmt.Errorf("failed to write to Python stdin: %w", err)
	}

	// Video generation timeout: 30 minutes (videos take much longer than images)
	select {
	case resp := <-respChan:
		if resp == nil {
			return nil, fmt.Errorf("received nil video response")
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("Python video error: %s", *resp.Error)
		}
		if resp.Result == nil {
			return nil, fmt.Errorf("received empty video result")
		}
		return resp.Result, nil

	case <-ctx.Done():
		return nil, fmt.Errorf("video request cancelled: %w", ctx.Err())

	case <-time.After(1800 * time.Second):
		return nil, fmt.Errorf("video generation timeout (30 minutes)")
	}
}

func (e *PythonExecutor) handleResponses() {
	for {
		select {
		case <-e.ctx.Done():
			return
		default:
		}

		line, err := e.stdout.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				log.Printf("Error reading from Python stdout: %v", err)
			}
			return
		}

		log.Printf("Received from Python: %s", line)

		// First, check if this is a progress message (has "type" field, no "id" field)
		// Progress messages are emitted during video generation and should be routed
		// to the progress callback without blocking response channels
		var msgType struct {
			Type string  `json:"type"`
			ID   *uint64 `json:"id"` // Pointer to detect absence vs zero
		}
		if err := json.Unmarshal([]byte(line), &msgType); err != nil {
			log.Printf("Failed to decode Python message: %v - %s", err, line)
			continue
		}

		// Route progress messages to callback (they have type="progress" and no id)
		if msgType.Type == "progress" && msgType.ID == nil {
			var progress models.ProgressMessage
			if err := json.Unmarshal([]byte(line), &progress); err != nil {
				log.Printf("Failed to decode progress message: %v - %s", err, line)
				continue
			}

			// Invoke callback if set (non-blocking)
			if cb := e.getProgressCallback(); cb != nil {
				// Call in goroutine to avoid blocking response processing
				go cb(progress)
			}
			continue // Don't try to route to response channels
		}

		// This is a final response (has an id field) - route to appropriate channel
		// Try to parse as a generic response to get the ID
		var baseResp struct {
			ID     uint64           `json:"id"`
			Result *json.RawMessage `json:"result"`
			Error  *string          `json:"error"`
		}
		if err := json.Unmarshal([]byte(line), &baseResp); err != nil {
			log.Printf("Failed to decode Python response: %v - %s", err, line)
			continue
		}

		// Check if this is a video response or image response based on response content
		e.mu.Lock()
		videoRespChan, isVideo := e.videoResponses[baseResp.ID]
		imageRespChan, isImage := e.responses[baseResp.ID]
		e.mu.Unlock()

		if isVideo {
			// Parse as video response
			var resp models.JSONRPCVideoResponse
			if err := json.Unmarshal([]byte(line), &resp); err != nil {
				log.Printf("Failed to decode video response: %v", err)
				continue
			}
			select {
			case videoRespChan <- &resp:
			default:
				log.Printf("Video response channel full for request ID: %d", resp.ID)
			}
		} else if isImage {
			// Parse as image response
			var resp models.JSONRPCResponse
			if err := json.Unmarshal([]byte(line), &resp); err != nil {
				log.Printf("Failed to decode image response: %v", err)
				continue
			}
			select {
			case imageRespChan <- &resp:
			default:
				log.Printf("Image response channel full for request ID: %d", resp.ID)
			}
		} else {
			log.Printf("Received response for unknown request ID: %d", baseResp.ID)
		}
	}
}

func (e *PythonExecutor) logStderr() {
	scanner := bufio.NewScanner(e.stderr)
	for scanner.Scan() {
		log.Printf("[Python stderr] %s", scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("Error reading Python stderr: %v", err)
	}
}

// Stop gracefully stops the Python process
func (e *PythonExecutor) Stop() error {
	log.Println("Stopping Python process...")

	e.cancel()

	if e.stdin != nil {
		e.stdin.Close()
	}

	done := make(chan error, 1)
	go func() {
		done <- e.cmd.Wait()
	}()

	select {
	case err := <-done:
		if err != nil {
			log.Printf("Python process exited with error: %v", err)
		} else {
			log.Println("Python process exited cleanly")
		}
		return err

	case <-time.After(10 * time.Second):
		log.Println("Python process didn't exit, killing...")
		if err := e.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill Python process: %w", err)
		}
		return fmt.Errorf("Python process killed after timeout")
	}
}

// SetProgressCallback sets the callback function for video progress updates.
// The callback is invoked each time Python emits a progress message during video generation.
// Set to nil to disable progress callbacks.
func (e *PythonExecutor) SetProgressCallback(cb ProgressCallback) {
	e.progressMu.Lock()
	defer e.progressMu.Unlock()
	e.progressCallback = cb
}

// getProgressCallback safely retrieves the current progress callback
func (e *PythonExecutor) getProgressCallback() ProgressCallback {
	e.progressMu.RLock()
	defer e.progressMu.RUnlock()
	return e.progressCallback
}
