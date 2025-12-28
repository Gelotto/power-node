package models

import "time"

// JobType represents the type of generation job
type JobType string

const (
	JobTypeImage JobType = "image"
	JobTypeVideo JobType = "video"
)

// Job represents an image or video generation job from the backend
type Job struct {
	ID             string  `json:"id"`
	Type           JobType `json:"type"` // "image" or "video"
	Status         string  `json:"status"`
	Prompt         string  `json:"prompt"`
	NegativePrompt *string `json:"negative_prompt,omitempty"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	Seed           *int64  `json:"seed,omitempty"`

	// Video-specific fields
	DurationSeconds *int `json:"duration_seconds,omitempty"`
	FPS             *int `json:"fps,omitempty"`
	TotalFrames     *int `json:"total_frames,omitempty"`
}

// IsVideoJob returns true if this is a video generation job
func (j *Job) IsVideoJob() bool {
	return j.Type == JobTypeVideo
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

// GenerateRequest is sent to Python subprocess for image generation
type GenerateRequest struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt *string `json:"negative_prompt,omitempty"`
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	Steps          int     `json:"steps"`
	Seed           *int64  `json:"seed,omitempty"`
}

// GenerateResponse is received from Python subprocess for image generation
type GenerateResponse struct {
	ImageData string `json:"image_data"`
	Format    string `json:"format"`
}

// GenerateVideoRequest is sent to Python subprocess for video generation
type GenerateVideoRequest struct {
	Prompt          string  `json:"prompt"`
	NegativePrompt  *string `json:"negative_prompt,omitempty"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	DurationSeconds int     `json:"duration_seconds"`
	FPS             int     `json:"fps"`
	TotalFrames     int     `json:"total_frames"`
	Seed            *int64  `json:"seed,omitempty"`
}

// GenerateVideoResponse is received from Python subprocess for video generation
type GenerateVideoResponse struct {
	VideoData       string `json:"video_data"` // Base64 encoded MP4
	Format          string `json:"format"`     // "mp4"
	FramesGenerated int    `json:"frames_generated"`
	DurationSeconds int    `json:"duration_seconds"`
}

// JSONRPCRequest is the JSON-RPC request format for image generation
type JSONRPCRequest struct {
	ID     uint64          `json:"id"`
	Method string          `json:"method"`
	Params GenerateRequest `json:"params"`
}

// JSONRPCVideoRequest is the JSON-RPC request format for video generation
type JSONRPCVideoRequest struct {
	ID     uint64               `json:"id"`
	Method string               `json:"method"`
	Params GenerateVideoRequest `json:"params"`
}

// JSONRPCResponse is the JSON-RPC response format for image generation
type JSONRPCResponse struct {
	ID     uint64            `json:"id"`
	Result *GenerateResponse `json:"result"`
	Error  *string           `json:"error"`
}

// JSONRPCVideoResponse is the JSON-RPC response format for video generation
type JSONRPCVideoResponse struct {
	ID     uint64                 `json:"id"`
	Result *GenerateVideoResponse `json:"result"`
	Error  *string                `json:"error"`
}

// ProgressMessage represents an intermediate progress update from Python during video generation
// These messages are emitted via stdout between request and final response
type ProgressMessage struct {
	Type            string  `json:"type"`             // Always "progress"
	Step            int     `json:"step"`             // Current denoising step (1-based)
	TotalSteps      int     `json:"total_steps"`      // Total denoising steps
	ProgressPercent float64 `json:"progress_percent"` // 0-100
	FramesCompleted int     `json:"frames_completed"` // Estimated frames based on step progress
}
