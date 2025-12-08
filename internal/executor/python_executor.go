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

// PythonExecutor manages the Python subprocess for image generation
type PythonExecutor struct {
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    *bufio.Reader
	stderr    io.ReadCloser
	reqID     uint64
	responses map[uint64]chan *models.JSONRPCResponse
	mu        sync.Mutex
	ctx       context.Context
	cancel    context.CancelFunc
}

// NewPythonExecutor creates a new Python executor
func NewPythonExecutor(pythonExec, scriptPath string, scriptArgs []string, env map[string]string) *PythonExecutor {
	ctx, cancel := context.WithCancel(context.Background())

	return &PythonExecutor{
		responses: make(map[uint64]chan *models.JSONRPCResponse),
		ctx:       ctx,
		cancel:    cancel,
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

		var resp models.JSONRPCResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			log.Printf("Failed to decode Python response: %v - %s", err, line)
			continue
		}

		e.mu.Lock()
		respChan, ok := e.responses[resp.ID]
		e.mu.Unlock()

		if !ok {
			log.Printf("Received response for unknown request ID: %d", resp.ID)
			continue
		}

		select {
		case respChan <- &resp:
		default:
			log.Printf("Response channel full for request ID: %d", resp.ID)
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
