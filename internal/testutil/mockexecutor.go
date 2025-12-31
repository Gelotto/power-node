package testutil

import (
	"context"
	"fmt"
	"sync"

	"github.com/Gelotto/power-node/internal/executor"
	"github.com/Gelotto/power-node/internal/models"
)

// MockExecutor is a mock implementation of PythonExecutor for testing
type MockExecutor struct {
	mu sync.Mutex

	// Configure behavior
	GenerateFunc        func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error)
	GenerateVideoFunc   func(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error)
	GenerateFaceSwapFunc func(ctx context.Context, req *models.FaceSwapRequest) (*models.FaceSwapResponse, error)
	StartFunc           func(pythonExec, scriptPath string, args []string, env map[string]string) error
	StopFunc            func() error

	// Track calls
	GenerateCalls        []models.GenerateRequest
	GenerateVideoCalls   []models.GenerateVideoRequest
	GenerateFaceSwapCalls []models.FaceSwapRequest
	StartCalls           int
	StopCalls            int

	// Progress callback
	progressCallback executor.ProgressCallback
}

// NewMockExecutor creates a new mock executor with default success behavior
func NewMockExecutor() *MockExecutor {
	return &MockExecutor{
		GenerateCalls:        make([]models.GenerateRequest, 0),
		GenerateVideoCalls:   make([]models.GenerateVideoRequest, 0),
		GenerateFaceSwapCalls: make([]models.FaceSwapRequest, 0),
	}
}

// Start mocks starting the Python process
func (m *MockExecutor) Start(pythonExec, scriptPath string, args []string, env map[string]string) error {
	m.mu.Lock()
	m.StartCalls++
	m.mu.Unlock()

	if m.StartFunc != nil {
		return m.StartFunc(pythonExec, scriptPath, args, env)
	}
	return nil
}

// Stop mocks stopping the Python process
func (m *MockExecutor) Stop() error {
	m.mu.Lock()
	m.StopCalls++
	m.mu.Unlock()

	if m.StopFunc != nil {
		return m.StopFunc()
	}
	return nil
}

// Generate mocks image generation
func (m *MockExecutor) Generate(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
	m.mu.Lock()
	m.GenerateCalls = append(m.GenerateCalls, *req)
	m.mu.Unlock()

	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, req)
	}

	// Default success response
	return &models.GenerateResponse{
		ImageData: "base64_mock_image_data",
		Format:    "png",
	}, nil
}

// GenerateVideo mocks video generation
func (m *MockExecutor) GenerateVideo(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error) {
	m.mu.Lock()
	m.GenerateVideoCalls = append(m.GenerateVideoCalls, *req)
	cb := m.progressCallback
	m.mu.Unlock()

	if m.GenerateVideoFunc != nil {
		return m.GenerateVideoFunc(ctx, req)
	}

	// Simulate progress callbacks if callback is set
	if cb != nil {
		cb(models.ProgressMessage{
			Type:            "progress",
			Step:            10,
			TotalSteps:      25,
			ProgressPercent: 40.0,
			FramesCompleted: 48,
		})
	}

	// Default success response
	frames := req.TotalFrames
	if frames == 0 {
		frames = 120
	}
	return &models.GenerateVideoResponse{
		VideoData:       "base64_mock_video_data",
		Format:          "mp4",
		FramesGenerated: frames,
		DurationSeconds: req.DurationSeconds,
	}, nil
}

// GenerateFaceSwap mocks face-swap generation
func (m *MockExecutor) GenerateFaceSwap(ctx context.Context, req *models.FaceSwapRequest) (*models.FaceSwapResponse, error) {
	m.mu.Lock()
	m.GenerateFaceSwapCalls = append(m.GenerateFaceSwapCalls, *req)
	m.mu.Unlock()

	if m.GenerateFaceSwapFunc != nil {
		return m.GenerateFaceSwapFunc(ctx, req)
	}

	// Default success response
	format := "png"
	if req.IsGIF {
		format = "gif"
	}
	return &models.FaceSwapResponse{
		ImageData:    "base64_mock_faceswap_data",
		Format:       format,
		FramesSwapped: 1,
	}, nil
}

// SetProgressCallback sets the progress callback
func (m *MockExecutor) SetProgressCallback(cb executor.ProgressCallback) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.progressCallback = cb
}

// SimulateProgress triggers progress callbacks for testing
func (m *MockExecutor) SimulateProgress(msg models.ProgressMessage) {
	m.mu.Lock()
	cb := m.progressCallback
	m.mu.Unlock()

	if cb != nil {
		cb(msg)
	}
}

// MockExecutorFailure configures the mock to fail with specific errors
type MockExecutorFailure struct {
	GenerateError        error
	GenerateVideoError   error
	GenerateFaceSwapError error
	StartError           error
	StopError            error
}

// ConfigureFailures sets up the mock to return specific errors
func (m *MockExecutor) ConfigureFailures(failures MockExecutorFailure) {
	if failures.GenerateError != nil {
		m.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
			return nil, failures.GenerateError
		}
	}
	if failures.GenerateVideoError != nil {
		m.GenerateVideoFunc = func(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error) {
			return nil, failures.GenerateVideoError
		}
	}
	if failures.GenerateFaceSwapError != nil {
		m.GenerateFaceSwapFunc = func(ctx context.Context, req *models.FaceSwapRequest) (*models.FaceSwapResponse, error) {
			return nil, failures.GenerateFaceSwapError
		}
	}
	if failures.StartError != nil {
		m.StartFunc = func(pythonExec, scriptPath string, args []string, env map[string]string) error {
			return failures.StartError
		}
	}
	if failures.StopError != nil {
		m.StopFunc = func() error {
			return failures.StopError
		}
	}
}

// Reset clears all recorded calls and resets to default behavior
func (m *MockExecutor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.GenerateCalls = make([]models.GenerateRequest, 0)
	m.GenerateVideoCalls = make([]models.GenerateVideoRequest, 0)
	m.GenerateFaceSwapCalls = make([]models.FaceSwapRequest, 0)
	m.StartCalls = 0
	m.StopCalls = 0
	m.GenerateFunc = nil
	m.GenerateVideoFunc = nil
	m.GenerateFaceSwapFunc = nil
	m.StartFunc = nil
	m.StopFunc = nil
	m.progressCallback = nil
}

// ContextCancelledError returns a context cancelled error for testing
func ContextCancelledError() error {
	return fmt.Errorf("context canceled")
}

// CUDAOutOfMemoryError returns a CUDA OOM error for testing
func CUDAOutOfMemoryError() error {
	return fmt.Errorf("CUDA out of memory")
}

// GenerationTimeoutError returns a generation timeout error for testing
func GenerationTimeoutError() error {
	return fmt.Errorf("generation timeout (5 minutes)")
}
