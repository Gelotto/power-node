package models

import "time"

// Job represents an image generation job from the backend
type Job struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Prompt string `json:"prompt"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Steps  int    `json:"steps"`
	Seed   *int64 `json:"seed,omitempty"`
}

// WorkerInfo represents worker registration/status
type WorkerInfo struct {
	ID            string    `json:"id"`
	Status        string    `json:"status"`
	Hostname      string    `json:"hostname"`
	ModelVersion  string    `json:"model_version"`
	JobsCompleted int       `json:"jobs_completed"`
	RegisteredAt  time.Time `json:"registered_at"`
}

// RegisterWorkerRequest is sent to register a new worker
type RegisterWorkerRequest struct {
	Hostname string `json:"hostname"`
}

// RegisterWorkerResponse is returned from worker registration
type RegisterWorkerResponse struct {
	APIKey    string    `json:"api_key"`
	KeyID     string    `json:"key_id"`
	WorkerID  string    `json:"worker_id"`
	Message   string    `json:"message"`
	CreatedAt time.Time `json:"created_at"`
}

// GenerateRequest is sent to Python subprocess
type GenerateRequest struct {
	Prompt string `json:"prompt"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Steps  int    `json:"steps"`
	Seed   *int64 `json:"seed,omitempty"`
}

// GenerateResponse is received from Python subprocess
type GenerateResponse struct {
	ImageData string `json:"image_data"`
	Format    string `json:"format"`
}

// JSONRPCRequest is the JSON-RPC request format
type JSONRPCRequest struct {
	ID     uint64          `json:"id"`
	Method string          `json:"method"`
	Params GenerateRequest `json:"params"`
}

// JSONRPCResponse is the JSON-RPC response format
type JSONRPCResponse struct {
	ID     uint64            `json:"id"`
	Result *GenerateResponse `json:"result"`
	Error  *string           `json:"error"`
}
