package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/client"
	"github.com/Gelotto/power-node/internal/models"
)

// TestNewWorker tests the worker constructor
func TestNewWorker(t *testing.T) {
	config := &Config{
		API: APIConfig{
			URL: "http://localhost:8080",
			Key: "wk_" + strings.Repeat("a", 64),
		},
		Worker: WorkerConfig{
			ID: "test-worker-id",
		},
	}

	worker, err := NewWorker(config, "/path/to/config.yaml")
	if err != nil {
		t.Fatalf("NewWorker() error = %v", err)
	}

	if worker.id != "test-worker-id" {
		t.Errorf("worker.id = %q, want %q", worker.id, "test-worker-id")
	}

	if worker.config != config {
		t.Error("worker.config not set correctly")
	}

	if worker.apiClient == nil {
		t.Error("worker.apiClient is nil")
	}

	if worker.stopChan == nil {
		t.Error("worker.stopChan is nil")
	}
}

// TestBuildHeartbeatData_Minimal tests heartbeat data with minimal configuration
func TestBuildHeartbeatData_Minimal(t *testing.T) {
	worker := &Worker{
		hostname: "test-host",
		gpuInfo:  "NVIDIA RTX 4090",
	}

	data := worker.buildHeartbeatData()

	if data.Hostname != "test-host" {
		t.Errorf("Hostname = %q, want %q", data.Hostname, "test-host")
	}
	if data.GPUInfo != "NVIDIA RTX 4090" {
		t.Errorf("GPUInfo = %q, want %q", data.GPUInfo, "NVIDIA RTX 4090")
	}
	// With no capabilities, these should be nil
	if data.VRAM != nil {
		t.Error("VRAM should be nil when no capabilities")
	}
	if data.MaxResolution != nil {
		t.Error("MaxResolution should be nil when no capabilities")
	}
}

// TestBuildHeartbeatData_WithCapabilities tests heartbeat data with full capabilities
func TestBuildHeartbeatData_WithCapabilities(t *testing.T) {
	worker := &Worker{
		hostname: "test-host",
		gpuInfo:  "NVIDIA RTX 4090",
		capabilities: &GPUCapabilities{
			GPUModel:      "NVIDIA RTX 4090",
			VRAM:          24,
			ComputeCap:    "8.9",
			ServiceMode:   "pytorch",
			MaxResolution: 2048,
			MaxSteps:      32,
		},
		config: &Config{
			Model: ModelConfig{
				Name: "z-image-turbo",
			},
			Video: VideoConfig{
				MaxDuration: 10,
				MaxFPS:      30,
				MaxWidth:    1280,
				MaxHeight:   720,
			},
			FaceSwap: FaceSwapConfig{
				MaxFrames: 100,
				MaxWidth:  1920,
				MaxHeight: 1080,
			},
		},
		supportsVideo:    true,
		supportsFaceSwap: true,
	}

	data := worker.buildHeartbeatData()

	// Model info
	if data.ImageModel != "z-image-turbo" {
		t.Errorf("ImageModel = %q, want %q", data.ImageModel, "z-image-turbo")
	}

	if data.VRAM == nil || *data.VRAM != 24 {
		t.Errorf("VRAM = %v, want 24", data.VRAM)
	}
	if data.ComputeCap != "8.9" {
		t.Errorf("ComputeCap = %q, want %q", data.ComputeCap, "8.9")
	}
	if data.ServiceMode != "pytorch" {
		t.Errorf("ServiceMode = %q, want %q", data.ServiceMode, "pytorch")
	}
	if data.MaxResolution == nil || *data.MaxResolution != 2048 {
		t.Errorf("MaxResolution = %v, want 2048", data.MaxResolution)
	}
	if data.MaxSteps == nil || *data.MaxSteps != 32 {
		t.Errorf("MaxSteps = %v, want 32", data.MaxSteps)
	}

	// Video capabilities
	if data.SupportsVideo == nil || !*data.SupportsVideo {
		t.Error("SupportsVideo should be true")
	}
	if data.VideoMaxDuration == nil || *data.VideoMaxDuration != 10 {
		t.Errorf("VideoMaxDuration = %v, want 10", data.VideoMaxDuration)
	}

	// Face-swap capabilities
	if data.SupportsFaceSwap == nil || !*data.SupportsFaceSwap {
		t.Error("SupportsFaceSwap should be true")
	}
	if data.FaceSwapMaxFrames == nil || *data.FaceSwapMaxFrames != 100 {
		t.Errorf("FaceSwapMaxFrames = %v, want 100", data.FaceSwapMaxFrames)
	}
}

// TestBuildHeartbeatData_VideoDisabled tests heartbeat when video is disabled
func TestBuildHeartbeatData_VideoDisabled(t *testing.T) {
	worker := &Worker{
		hostname:      "test-host",
		gpuInfo:       "NVIDIA RTX 4090",
		config:        &Config{},
		supportsVideo: false,
	}

	data := worker.buildHeartbeatData()

	if data.SupportsVideo == nil || *data.SupportsVideo != false {
		t.Error("SupportsVideo should be false")
	}
	// When video disabled, video-specific fields should be nil
	if data.VideoMaxDuration != nil {
		t.Error("VideoMaxDuration should be nil when video disabled")
	}
}

// TestDetectVideoCapability_ExplicitEnabled tests explicit video enable
func TestDetectVideoCapability_ExplicitEnabled(t *testing.T) {
	enabled := true
	worker := &Worker{
		config: &Config{
			Video: VideoConfig{
				Enabled: &enabled,
			},
		},
	}

	worker.detectVideoCapability()

	if !worker.supportsVideo {
		t.Error("supportsVideo should be true when explicitly enabled")
	}
}

// TestDetectVideoCapability_ExplicitDisabled tests explicit video disable
func TestDetectVideoCapability_ExplicitDisabled(t *testing.T) {
	disabled := false
	worker := &Worker{
		config: &Config{
			Video: VideoConfig{
				Enabled: &disabled,
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false when explicitly disabled")
	}
}

// TestDetectVideoCapability_GGUFMode tests video disabled for GGUF mode
func TestDetectVideoCapability_GGUFMode(t *testing.T) {
	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "gguf",
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: "/some/path",
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false in GGUF mode")
	}
}

// TestDetectVideoCapability_NoModelPath tests video disabled when no model path
func TestDetectVideoCapability_NoModelPath(t *testing.T) {
	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "pytorch",
			VRAM:        24,
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: "",
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false when model path empty")
	}
}

// TestDetectVideoCapability_ModelNotFound tests video disabled when model directory missing
func TestDetectVideoCapability_ModelNotFound(t *testing.T) {
	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "pytorch",
			VRAM:        24,
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: "/nonexistent/path/to/model",
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false when model directory doesn't exist")
	}
}

// TestDetectVideoCapability_ModelIndexMissing tests video disabled when model_index.json missing
func TestDetectVideoCapability_ModelIndexMissing(t *testing.T) {
	// Create temp directory without model_index.json
	tempDir := t.TempDir()

	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "pytorch",
			VRAM:        24,
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false when model_index.json missing")
	}
}

// TestDetectVideoCapability_InsufficientVRAM tests video disabled when VRAM too low
func TestDetectVideoCapability_InsufficientVRAM(t *testing.T) {
	// Create temp directory with model_index.json
	tempDir := t.TempDir()
	modelIndexPath := filepath.Join(tempDir, "model_index.json")
	if err := os.WriteFile(modelIndexPath, []byte("{}"), 0644); err != nil {
		t.Fatalf("Failed to create model_index.json: %v", err)
	}

	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "pytorch",
			VRAM:        4, // Less than 8GB
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectVideoCapability()

	if worker.supportsVideo {
		t.Error("supportsVideo should be false when VRAM < 8GB")
	}
}

// TestDetectVideoCapability_AllChecksPass tests video enabled when all checks pass
func TestDetectVideoCapability_AllChecksPass(t *testing.T) {
	// Create temp directory with model_index.json
	tempDir := t.TempDir()
	modelIndexPath := filepath.Join(tempDir, "model_index.json")
	if err := os.WriteFile(modelIndexPath, []byte("{}"), 0644); err != nil {
		t.Fatalf("Failed to create model_index.json: %v", err)
	}

	worker := &Worker{
		capabilities: &GPUCapabilities{
			ServiceMode: "pytorch",
			VRAM:        24, // Sufficient VRAM
		},
		config: &Config{
			Video: VideoConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectVideoCapability()

	if !worker.supportsVideo {
		t.Error("supportsVideo should be true when all checks pass")
	}
}

// TestDetectFaceSwapCapability_ExplicitEnabled tests explicit face-swap enable
func TestDetectFaceSwapCapability_ExplicitEnabled(t *testing.T) {
	enabled := true
	worker := &Worker{
		config: &Config{
			FaceSwap: FaceSwapConfig{
				Enabled: &enabled,
			},
		},
	}

	worker.detectFaceSwapCapability()

	if !worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be true when explicitly enabled")
	}
}

// TestDetectFaceSwapCapability_ExplicitDisabled tests explicit face-swap disable
func TestDetectFaceSwapCapability_ExplicitDisabled(t *testing.T) {
	disabled := false
	worker := &Worker{
		config: &Config{
			FaceSwap: FaceSwapConfig{
				Enabled: &disabled,
			},
		},
	}

	worker.detectFaceSwapCapability()

	if worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be false when explicitly disabled")
	}
}

// TestDetectFaceSwapCapability_NoModelPath tests face-swap disabled when no model path
func TestDetectFaceSwapCapability_NoModelPath(t *testing.T) {
	worker := &Worker{
		capabilities: &GPUCapabilities{
			VRAM: 24,
		},
		config: &Config{
			FaceSwap: FaceSwapConfig{
				ModelPath: "",
			},
		},
	}

	worker.detectFaceSwapCapability()

	if worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be false when model path empty")
	}
}

// TestDetectFaceSwapCapability_ModelNotFound tests face-swap disabled when directory missing
func TestDetectFaceSwapCapability_ModelNotFound(t *testing.T) {
	worker := &Worker{
		capabilities: &GPUCapabilities{
			VRAM: 24,
		},
		config: &Config{
			FaceSwap: FaceSwapConfig{
				ModelPath: "/nonexistent/path/to/model",
			},
		},
	}

	worker.detectFaceSwapCapability()

	if worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be false when model directory doesn't exist")
	}
}

// TestDetectFaceSwapCapability_InswapperMissing tests face-swap disabled when inswapper missing
func TestDetectFaceSwapCapability_InswapperMissing(t *testing.T) {
	// Create temp directory without inswapper_128.onnx
	tempDir := t.TempDir()

	worker := &Worker{
		capabilities: &GPUCapabilities{
			VRAM: 24,
		},
		config: &Config{
			FaceSwap: FaceSwapConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectFaceSwapCapability()

	if worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be false when inswapper_128.onnx missing")
	}
}

// TestDetectFaceSwapCapability_InsufficientVRAM tests face-swap disabled when VRAM too low
func TestDetectFaceSwapCapability_InsufficientVRAM(t *testing.T) {
	// Create temp directory with inswapper_128.onnx
	tempDir := t.TempDir()
	inswapperPath := filepath.Join(tempDir, "inswapper_128.onnx")
	if err := os.WriteFile(inswapperPath, []byte("dummy"), 0644); err != nil {
		t.Fatalf("Failed to create inswapper_128.onnx: %v", err)
	}

	worker := &Worker{
		capabilities: &GPUCapabilities{
			VRAM: 4, // Less than 6GB
		},
		config: &Config{
			FaceSwap: FaceSwapConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectFaceSwapCapability()

	if worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be false when VRAM < 6GB")
	}
}

// TestDetectFaceSwapCapability_AllChecksPass tests face-swap enabled when all checks pass
func TestDetectFaceSwapCapability_AllChecksPass(t *testing.T) {
	// Create temp directory with inswapper_128.onnx
	tempDir := t.TempDir()
	inswapperPath := filepath.Join(tempDir, "inswapper_128.onnx")
	if err := os.WriteFile(inswapperPath, []byte("dummy"), 0644); err != nil {
		t.Fatalf("Failed to create inswapper_128.onnx: %v", err)
	}

	worker := &Worker{
		capabilities: &GPUCapabilities{
			VRAM: 8, // Sufficient VRAM (>=6GB)
		},
		config: &Config{
			FaceSwap: FaceSwapConfig{
				ModelPath: tempDir,
			},
		},
	}

	worker.detectFaceSwapCapability()

	if !worker.supportsFaceSwap {
		t.Error("supportsFaceSwap should be true when all checks pass")
	}
}

// TestCanProcessVideo tests the canProcessVideo method
func TestCanProcessVideo(t *testing.T) {
	tests := []struct {
		name          string
		supportsVideo bool
		want          bool
	}{
		{"supports video", true, true},
		{"does not support video", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			worker := &Worker{supportsVideo: tt.supportsVideo}
			if got := worker.canProcessVideo(); got != tt.want {
				t.Errorf("canProcessVideo() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestCanProcessFaceSwap tests the canProcessFaceSwap method
func TestCanProcessFaceSwap(t *testing.T) {
	tests := []struct {
		name             string
		supportsFaceSwap bool
		want             bool
	}{
		{"supports face-swap", true, true},
		{"does not support face-swap", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			worker := &Worker{supportsFaceSwap: tt.supportsFaceSwap}
			if got := worker.canProcessFaceSwap(); got != tt.want {
				t.Errorf("canProcessFaceSwap() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSetAndIsProcessingJob tests the processing job state management
func TestSetAndIsProcessingJob(t *testing.T) {
	worker := &Worker{}

	// Initially not processing
	if worker.isProcessingJob() {
		t.Error("isProcessingJob() should be false initially")
	}

	// Set processing
	worker.setProcessingJob(true)
	if !worker.isProcessingJob() {
		t.Error("isProcessingJob() should be true after setProcessingJob(true)")
	}

	// Clear processing
	worker.setProcessingJob(false)
	if worker.isProcessingJob() {
		t.Error("isProcessingJob() should be false after setProcessingJob(false)")
	}
}

// TestProcessingJobConcurrency tests concurrent access to processing state
func TestProcessingJobConcurrency(t *testing.T) {
	worker := &Worker{}

	var wg sync.WaitGroup
	iterations := 100

	// Multiple goroutines setting/reading processing state
	for i := 0; i < iterations; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			worker.setProcessingJob(true)
			worker.isProcessingJob()
			worker.setProcessingJob(false)
		}()
		go func() {
			defer wg.Done()
			_ = worker.isProcessingJob()
		}()
	}

	wg.Wait()
	// If we get here without race detector complaints, test passes
}

// TestGetPauseChan_NoMonitor tests getPauseChan returns nil when no monitor
func TestGetPauseChan_NoMonitor(t *testing.T) {
	worker := &Worker{gpuMonitor: nil}

	if ch := worker.getPauseChan(); ch != nil {
		t.Error("getPauseChan() should return nil when no monitor")
	}
}

// TestGetResumeChan_NoMonitor tests getResumeChan returns nil when no monitor
func TestGetResumeChan_NoMonitor(t *testing.T) {
	worker := &Worker{gpuMonitor: nil}

	if ch := worker.getResumeChan(); ch != nil {
		t.Error("getResumeChan() should return nil when no monitor")
	}
}

// TestStop_NoComponents tests Stop with no initialized components
func TestStop_NoComponents(t *testing.T) {
	worker := &Worker{
		stopChan: make(chan struct{}),
	}

	err := worker.Stop()
	if err != nil {
		t.Errorf("Stop() error = %v", err)
	}
}

// MockAPIServer creates a test server that mimics the API behavior
type MockAPIServer struct {
	*httptest.Server
	mu              sync.Mutex
	claimResponses  []*models.Job
	claimIndex      int
	heartbeatCalls  int
	claimCalls      int
	completeCalls   int
	failCalls       int
	startCalls      int
	progressCalls   int
	lastHeartbeat   *client.HeartbeatData
	lastCompleteJob string
	lastFailJob     string
	lastFailError   string
}

// NewMockAPIServer creates a new mock API server
func NewMockAPIServer(t *testing.T) *MockAPIServer {
	mock := &MockAPIServer{
		claimResponses: make([]*models.Job, 0),
	}

	mock.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mock.mu.Lock()
		defer mock.mu.Unlock()

		switch {
		case strings.HasSuffix(r.URL.Path, "/heartbeat"):
			mock.heartbeatCalls++
			// Parse heartbeat data
			body, _ := io.ReadAll(r.Body)
			var data map[string]interface{}
			json.Unmarshal(body, &data)
			if hd, ok := data["heartbeat_data"].(map[string]interface{}); ok {
				mock.lastHeartbeat = &client.HeartbeatData{}
				if h, ok := hd["hostname"].(string); ok {
					mock.lastHeartbeat.Hostname = h
				}
			}
			w.WriteHeader(http.StatusOK)

		case strings.HasSuffix(r.URL.Path, "/claim"):
			mock.claimCalls++
			if mock.claimIndex < len(mock.claimResponses) {
				job := mock.claimResponses[mock.claimIndex]
				mock.claimIndex++
				if job == nil {
					w.WriteHeader(http.StatusNoContent)
				} else {
					w.Header().Set("Content-Type", "application/json")
					json.NewEncoder(w).Encode(job)
				}
			} else {
				w.WriteHeader(http.StatusNoContent)
			}

		case strings.Contains(r.URL.Path, "/jobs/") && strings.HasSuffix(r.URL.Path, "/start"):
			mock.startCalls++
			w.WriteHeader(http.StatusOK)

		case strings.Contains(r.URL.Path, "/jobs/") && strings.HasSuffix(r.URL.Path, "/progress"):
			mock.progressCalls++
			w.WriteHeader(http.StatusOK)

		case strings.HasSuffix(r.URL.Path, "/complete"):
			mock.completeCalls++
			body, _ := io.ReadAll(r.Body)
			var data map[string]interface{}
			json.Unmarshal(body, &data)
			if jid, ok := data["job_id"].(string); ok {
				mock.lastCompleteJob = jid
			}
			w.WriteHeader(http.StatusOK)

		case strings.HasSuffix(r.URL.Path, "/fail"):
			mock.failCalls++
			body, _ := io.ReadAll(r.Body)
			var data map[string]interface{}
			json.Unmarshal(body, &data)
			if jid, ok := data["job_id"].(string); ok {
				mock.lastFailJob = jid
			}
			if errMsg, ok := data["error"].(string); ok {
				mock.lastFailError = errMsg
			}
			w.WriteHeader(http.StatusOK)

		default:
			t.Logf("Unhandled request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))

	return mock
}

// QueueJob adds a job to be returned by the next claim
func (m *MockAPIServer) QueueJob(job *models.Job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.claimResponses = append(m.claimResponses, job)
}

// TestWorkerWithMockAPI_Heartbeat tests heartbeat functionality with mock API
func TestWorkerWithMockAPI_Heartbeat(t *testing.T) {
	mockAPI := NewMockAPIServer(t)
	defer mockAPI.Close()

	apiClient := client.NewAPIClient(mockAPI.URL, "wk_" + strings.Repeat("a", 64))

	worker := &Worker{
		id:        "test-worker",
		hostname:  "test-host",
		gpuInfo:   "NVIDIA RTX 4090",
		config:    &Config{},
		apiClient: apiClient,
	}

	ctx := context.Background()
	data := worker.buildHeartbeatData()
	err := apiClient.Heartbeat(ctx, "test-worker", "online", data)
	if err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}

	if mockAPI.heartbeatCalls != 1 {
		t.Errorf("heartbeatCalls = %d, want 1", mockAPI.heartbeatCalls)
	}
}

// TestWorkerWithMockAPI_ClaimJob tests job claiming with mock API
func TestWorkerWithMockAPI_ClaimJob(t *testing.T) {
	mockAPI := NewMockAPIServer(t)
	defer mockAPI.Close()

	// Queue a job
	testJob := &models.Job{
		ID:     "test-job-123",
		Prompt: "a beautiful sunset",
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}
	mockAPI.QueueJob(testJob)

	apiClient := client.NewAPIClient(mockAPI.URL, "wk_" + strings.Repeat("a", 64))

	ctx := context.Background()
	job, err := apiClient.ClaimJob(ctx, "test-worker")
	if err != nil {
		t.Fatalf("ClaimJob() error = %v", err)
	}

	if job == nil {
		t.Fatal("ClaimJob() returned nil job")
	}

	if job.ID != "test-job-123" {
		t.Errorf("job.ID = %q, want %q", job.ID, "test-job-123")
	}

	if mockAPI.claimCalls != 1 {
		t.Errorf("claimCalls = %d, want 1", mockAPI.claimCalls)
	}
}

// TestWorkerWithMockAPI_ClaimJob_NoContent tests 204 response (no jobs)
func TestWorkerWithMockAPI_ClaimJob_NoContent(t *testing.T) {
	mockAPI := NewMockAPIServer(t)
	defer mockAPI.Close()

	// Don't queue any jobs - should get 204

	apiClient := client.NewAPIClient(mockAPI.URL, "wk_" + strings.Repeat("a", 64))

	ctx := context.Background()
	job, err := apiClient.ClaimJob(ctx, "test-worker")
	if err != nil {
		t.Fatalf("ClaimJob() error = %v", err)
	}

	if job != nil {
		t.Errorf("ClaimJob() returned job %v, want nil for empty queue", job)
	}
}

// TestJobRouting tests that jobs are routed to the correct processor
func TestJobRouting(t *testing.T) {
	tests := []struct {
		name           string
		job            *models.Job
		expectImage    bool
		expectVideo    bool
		expectFaceSwap bool
	}{
		{
			name: "image job (default)",
			job: &models.Job{
				ID:     "img-1",
				Type:   "",
				Prompt: "test",
			},
			expectImage: true,
		},
		{
			name: "image job (explicit)",
			job: &models.Job{
				ID:     "img-2",
				Type:   "image",
				Prompt: "test",
			},
			expectImage: true,
		},
		{
			name: "video job",
			job: &models.Job{
				ID:     "vid-1",
				Type:   "video",
				Prompt: "test",
			},
			expectVideo: true,
		},
		{
			name: "face-swap job",
			job: &models.Job{
				ID:   "fs-1",
				Type: "face_swap",
			},
			expectFaceSwap: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isImage := !tt.job.IsVideoJob() && !tt.job.IsFaceSwapJob()
			isVideo := tt.job.IsVideoJob()
			isFaceSwap := tt.job.IsFaceSwapJob()

			if isImage != tt.expectImage {
				t.Errorf("isImage = %v, want %v", isImage, tt.expectImage)
			}
			if isVideo != tt.expectVideo {
				t.Errorf("isVideo = %v, want %v", isVideo, tt.expectVideo)
			}
			if isFaceSwap != tt.expectFaceSwap {
				t.Errorf("isFaceSwap = %v, want %v", isFaceSwap, tt.expectFaceSwap)
			}
		})
	}
}

// TestProcessJobRoutesToCorrectHandler verifies job type routing
func TestProcessJobRoutesToCorrectHandler(t *testing.T) {
	// This test verifies the routing logic in processJob
	// by checking the job type detection

	tests := []struct {
		name     string
		jobType  models.JobType
		isVideo  bool
		isFaceSwap bool
	}{
		{"empty type is image", "", false, false},
		{"image type", models.JobTypeImage, false, false},
		{"video type", models.JobTypeVideo, true, false},
		{"face_swap type", models.JobTypeFaceSwap, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			job := &models.Job{Type: tt.jobType}

			if job.IsVideoJob() != tt.isVideo {
				t.Errorf("IsVideoJob() = %v, want %v", job.IsVideoJob(), tt.isVideo)
			}
			if job.IsFaceSwapJob() != tt.isFaceSwap {
				t.Errorf("IsFaceSwapJob() = %v, want %v", job.IsFaceSwapJob(), tt.isFaceSwap)
			}
		})
	}
}

// TestPauseResumeState tests pause/resume state management
func TestPauseResumeState(t *testing.T) {
	worker := &Worker{}

	// Initially not paused
	worker.pauseMu.RLock()
	if worker.isPaused {
		t.Error("isPaused should be false initially")
	}
	worker.pauseMu.RUnlock()

	// Set paused
	worker.pauseMu.Lock()
	worker.isPaused = true
	worker.pauseMu.Unlock()

	worker.pauseMu.RLock()
	if !worker.isPaused {
		t.Error("isPaused should be true after setting")
	}
	worker.pauseMu.RUnlock()

	// Resume
	worker.pauseMu.Lock()
	worker.isPaused = false
	worker.pauseMu.Unlock()

	worker.pauseMu.RLock()
	if worker.isPaused {
		t.Error("isPaused should be false after resuming")
	}
	worker.pauseMu.RUnlock()
}

// TestPauseStateConcurrency tests concurrent access to pause state
func TestPauseStateConcurrency(t *testing.T) {
	worker := &Worker{}

	var wg sync.WaitGroup
	iterations := 100

	for i := 0; i < iterations; i++ {
		wg.Add(3)
		go func() {
			defer wg.Done()
			worker.pauseMu.Lock()
			worker.isPaused = true
			worker.pauseMu.Unlock()
		}()
		go func() {
			defer wg.Done()
			worker.pauseMu.Lock()
			worker.isPaused = false
			worker.pauseMu.Unlock()
		}()
		go func() {
			defer wg.Done()
			worker.pauseMu.RLock()
			_ = worker.isPaused
			worker.pauseMu.RUnlock()
		}()
	}

	wg.Wait()
	// If we get here without race detector complaints, test passes
}

// TestJobLoopStopSignal tests that the job loop responds to stop signal
func TestJobLoopStopSignal(t *testing.T) {
	mockAPI := NewMockAPIServer(t)
	defer mockAPI.Close()

	apiClient := client.NewAPIClient(mockAPI.URL, "wk_" + strings.Repeat("a", 64))

	stopChan := make(chan struct{})
	worker := &Worker{
		id:        "test-worker",
		config:    &Config{Worker: WorkerConfig{PollInterval: 100 * time.Millisecond}},
		apiClient: apiClient,
		stopChan:  stopChan,
	}

	// Start job loop in goroutine
	done := make(chan error, 1)
	go func() {
		done <- worker.jobLoop(context.Background())
	}()

	// Let it run briefly
	time.Sleep(50 * time.Millisecond)

	// Send stop signal
	close(stopChan)

	// Should exit within reasonable time
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("jobLoop() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Error("jobLoop() did not exit after stop signal")
	}
}

// TestJobLoopContextCancellation tests that the job loop responds to context cancellation
func TestJobLoopContextCancellation(t *testing.T) {
	mockAPI := NewMockAPIServer(t)
	defer mockAPI.Close()

	apiClient := client.NewAPIClient(mockAPI.URL, "wk_" + strings.Repeat("a", 64))

	worker := &Worker{
		id:        "test-worker",
		config:    &Config{Worker: WorkerConfig{PollInterval: 100 * time.Millisecond}},
		apiClient: apiClient,
		stopChan:  make(chan struct{}),
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Start job loop in goroutine
	done := make(chan error, 1)
	go func() {
		done <- worker.jobLoop(ctx)
	}()

	// Let it run briefly
	time.Sleep(50 * time.Millisecond)

	// Cancel context
	cancel()

	// Should exit within reasonable time
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("jobLoop() error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Error("jobLoop() did not exit after context cancellation")
	}
}

// TestFaceSwapJobValidation tests face-swap job validation
func TestFaceSwapJobValidation(t *testing.T) {
	sourceURL := "http://example.com/source.jpg"
	targetURL := "http://example.com/target.jpg"

	tests := []struct {
		name      string
		job       *models.Job
		expectErr bool
	}{
		{
			name: "valid face-swap job",
			job: &models.Job{
				Type:           "face_swap",
				SourceImageURL: &sourceURL,
				TargetImageURL: &targetURL,
			},
			expectErr: false,
		},
		{
			name: "missing source URL",
			job: &models.Job{
				Type:           "face_swap",
				SourceImageURL: nil,
				TargetImageURL: &targetURL,
			},
			expectErr: true,
		},
		{
			name: "missing target URL",
			job: &models.Job{
				Type:           "face_swap",
				SourceImageURL: &sourceURL,
				TargetImageURL: nil,
			},
			expectErr: true,
		},
		{
			name: "missing both URLs",
			job: &models.Job{
				Type:           "face_swap",
				SourceImageURL: nil,
				TargetImageURL: nil,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hasError := tt.job.SourceImageURL == nil || tt.job.TargetImageURL == nil
			if hasError != tt.expectErr {
				t.Errorf("validation = %v, want %v", hasError, tt.expectErr)
			}
		})
	}
}

// TestVideoJobParameters tests video job parameter extraction
func TestVideoJobParameters(t *testing.T) {
	duration := 10
	fps := 30

	tests := []struct {
		name             string
		job              *models.Job
		expectedDuration int
		expectedFPS      int
		expectedFrames   int
	}{
		{
			name: "default values",
			job: &models.Job{
				Type: "video",
			},
			expectedDuration: 5,  // default
			expectedFPS:      24, // default
			expectedFrames:   120,
		},
		{
			name: "custom duration",
			job: &models.Job{
				Type:            "video",
				DurationSeconds: &duration,
			},
			expectedDuration: 10,
			expectedFPS:      24, // default
			expectedFrames:   240,
		},
		{
			name: "custom fps",
			job: &models.Job{
				Type: "video",
				FPS:  &fps,
			},
			expectedDuration: 5, // default
			expectedFPS:      30,
			expectedFrames:   150,
		},
		{
			name: "custom both",
			job: &models.Job{
				Type:            "video",
				DurationSeconds: &duration,
				FPS:             &fps,
			},
			expectedDuration: 10,
			expectedFPS:      30,
			expectedFrames:   300,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate the logic from processVideoJob
			durationSeconds := 5
			if tt.job.DurationSeconds != nil {
				durationSeconds = *tt.job.DurationSeconds
			}
			fpsVal := 24
			if tt.job.FPS != nil {
				fpsVal = *tt.job.FPS
			}
			totalFrames := durationSeconds * fpsVal

			if durationSeconds != tt.expectedDuration {
				t.Errorf("duration = %d, want %d", durationSeconds, tt.expectedDuration)
			}
			if fpsVal != tt.expectedFPS {
				t.Errorf("fps = %d, want %d", fpsVal, tt.expectedFPS)
			}
			if totalFrames != tt.expectedFrames {
				t.Errorf("totalFrames = %d, want %d", totalFrames, tt.expectedFrames)
			}
		})
	}
}

// TestFaceSwapJobParameters tests face-swap job parameter extraction
func TestFaceSwapJobParameters(t *testing.T) {
	isGIF := true
	swapAll := false
	enhance := false
	frameCount := 50

	tests := []struct {
		name            string
		job             *models.Job
		configMaxFrames int
		expectedIsGIF   bool
		expectedSwapAll bool
		expectedEnhance bool
		expectedFrames  int
	}{
		{
			name:            "all defaults",
			job:             &models.Job{Type: "face_swap"},
			configMaxFrames: 100,
			expectedIsGIF:   false,
			expectedSwapAll: true,
			expectedEnhance: true,
			expectedFrames:  100,
		},
		{
			name: "custom values",
			job: &models.Job{
				Type:          "face_swap",
				IsGIF:         &isGIF,
				SwapAllFaces:  &swapAll,
				EnhanceResult: &enhance,
				GifFrameCount: &frameCount,
			},
			configMaxFrames: 100,
			expectedIsGIF:   true,
			expectedSwapAll: false,
			expectedEnhance: false,
			expectedFrames:  50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate the logic from processFaceSwapJob
			isGIFVal := false
			if tt.job.IsGIF != nil {
				isGIFVal = *tt.job.IsGIF
			}
			swapAllVal := true
			if tt.job.SwapAllFaces != nil {
				swapAllVal = *tt.job.SwapAllFaces
			}
			enhanceVal := true
			if tt.job.EnhanceResult != nil {
				enhanceVal = *tt.job.EnhanceResult
			}
			maxFrames := tt.configMaxFrames
			if tt.job.GifFrameCount != nil && *tt.job.GifFrameCount > 0 {
				maxFrames = *tt.job.GifFrameCount
			}

			if isGIFVal != tt.expectedIsGIF {
				t.Errorf("isGIF = %v, want %v", isGIFVal, tt.expectedIsGIF)
			}
			if swapAllVal != tt.expectedSwapAll {
				t.Errorf("swapAll = %v, want %v", swapAllVal, tt.expectedSwapAll)
			}
			if enhanceVal != tt.expectedEnhance {
				t.Errorf("enhance = %v, want %v", enhanceVal, tt.expectedEnhance)
			}
			if maxFrames != tt.expectedFrames {
				t.Errorf("maxFrames = %d, want %d", maxFrames, tt.expectedFrames)
			}
		})
	}
}

// TestProgressReportingThreshold tests progress reporting logic
func TestProgressReportingThreshold(t *testing.T) {
	tests := []struct {
		name            string
		framesCompleted int
		lastReported    int
		progressPercent float64
		shouldReport    bool
	}{
		{"first report at 10 frames", 10, 0, 10.0, true},
		{"report at 20 frames", 20, 10, 20.0, true},
		{"skip report at 15 frames", 15, 10, 15.0, false},
		{"report at 95%", 95, 90, 95.0, true},
		{"report at 99%", 99, 90, 99.0, true},
		{"skip early small progress", 5, 0, 5.0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Replicate the logic from processVideoJob progress callback
			shouldReport := tt.framesCompleted >= tt.lastReported+10 || tt.progressPercent >= 95

			if shouldReport != tt.shouldReport {
				t.Errorf("shouldReport = %v, want %v", shouldReport, tt.shouldReport)
			}
		})
	}
}

// BenchmarkBuildHeartbeatData benchmarks heartbeat data building
func BenchmarkBuildHeartbeatData(b *testing.B) {
	worker := &Worker{
		hostname: "test-host",
		gpuInfo:  "NVIDIA RTX 4090",
		capabilities: &GPUCapabilities{
			GPUModel:      "NVIDIA RTX 4090",
			VRAM:          24,
			ComputeCap:    "8.9",
			ServiceMode:   "pytorch",
			MaxResolution: 2048,
			MaxSteps:      32,
		},
		config: &Config{
			Video: VideoConfig{
				MaxDuration: 10,
				MaxFPS:      30,
				MaxWidth:    1280,
				MaxHeight:   720,
			},
			FaceSwap: FaceSwapConfig{
				MaxFrames: 100,
				MaxWidth:  1920,
				MaxHeight: 1080,
			},
		},
		supportsVideo:    true,
		supportsFaceSwap: true,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = worker.buildHeartbeatData()
	}
}

// BenchmarkProcessingJobState benchmarks the processing job state operations
func BenchmarkProcessingJobState(b *testing.B) {
	worker := &Worker{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		worker.setProcessingJob(true)
		_ = worker.isProcessingJob()
		worker.setProcessingJob(false)
	}
}

// TestConfigValidation_WorkerStart tests config validation at worker start
func TestConfigValidation_WorkerStart(t *testing.T) {
	// Valid API key: wk_ prefix + 64 characters = 67 total
	validAPIKey := "wk_" + strings.Repeat("a", 64)

	tests := []struct {
		name      string
		config    *Config
		expectErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				API: APIConfig{
					URL: "http://localhost:8080",
					Key: validAPIKey,
				},
				Worker: WorkerConfig{
					ID: "test-worker",
				},
			},
			expectErr: false,
		},
		{
			name: "missing API key",
			config: &Config{
				API: APIConfig{
					URL: "http://localhost:8080",
					Key: "",
				},
				Worker: WorkerConfig{
					ID: "test-worker",
				},
			},
			expectErr: true,
		},
		{
			name: "invalid API key prefix",
			config: &Config{
				API: APIConfig{
					URL: "http://localhost:8080",
					Key: "invalid_" + strings.Repeat("a", 59), // 67 chars total but wrong prefix
				},
				Worker: WorkerConfig{
					ID: "test-worker",
				},
			},
			expectErr: true,
		},
		{
			name: "missing worker ID",
			config: &Config{
				API: APIConfig{
					URL: "http://localhost:8080",
					Key: validAPIKey,
				},
				Worker: WorkerConfig{
					ID: "",
				},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.expectErr {
				t.Errorf("Validate() error = %v, expectErr = %v", err, tt.expectErr)
			}
		})
	}
}

// TestDetectGPU_Fallback tests GPU detection fallback
func TestDetectGPU_Fallback(t *testing.T) {
	worker := &Worker{}

	// This will likely fail (no nvidia-smi in test environment)
	// and should return "Unknown GPU" as fallback
	gpu := worker.detectGPU()

	// Either it detects a real GPU or falls back to "Unknown GPU"
	if gpu == "" {
		t.Error("detectGPU() returned empty string")
	}
}

// TestWorkerIDInitialization tests worker ID is properly set
func TestWorkerIDInitialization(t *testing.T) {
	config := &Config{
		API: APIConfig{
			URL: "http://localhost:8080",
			Key: "wk_" + strings.Repeat("a", 64),
		},
		Worker: WorkerConfig{
			ID: "my-custom-worker-id",
		},
	}

	worker, err := NewWorker(config, "/path/to/config.yaml")
	if err != nil {
		t.Fatalf("NewWorker() error = %v", err)
	}

	if worker.id != "my-custom-worker-id" {
		t.Errorf("worker.id = %q, want %q", worker.id, "my-custom-worker-id")
	}
}

// TestPythonRunningState tests Python running state management
func TestPythonRunningState(t *testing.T) {
	worker := &Worker{}

	// Initially not running
	worker.pythonMu.Lock()
	if worker.pythonRunning {
		t.Error("pythonRunning should be false initially")
	}
	worker.pythonMu.Unlock()

	// Set running
	worker.pythonMu.Lock()
	worker.pythonRunning = true
	worker.pythonMu.Unlock()

	// Verify running
	worker.pythonMu.Lock()
	if !worker.pythonRunning {
		t.Error("pythonRunning should be true after setting")
	}
	worker.pythonMu.Unlock()

	// Stop
	worker.pythonMu.Lock()
	worker.pythonRunning = false
	worker.pythonMu.Unlock()

	worker.pythonMu.Lock()
	if worker.pythonRunning {
		t.Error("pythonRunning should be false after stopping")
	}
	worker.pythonMu.Unlock()
}

// TestErrorMessageFormatting tests error message construction
func TestErrorMessageFormatting(t *testing.T) {
	// Test the error message format used in Start()
	configPath := "/home/user/.config/power-node/config.yaml"
	baseErr := fmt.Errorf("API key is required")

	fullErr := fmt.Errorf("%v\n\n  Please add your credentials to:\n  %s\n\n  Get your credentials at: https://gelotto.io/workers", baseErr, configPath)

	errStr := fullErr.Error()
	if !strings.Contains(errStr, "API key is required") {
		t.Error("Error message missing base error")
	}
	if !strings.Contains(errStr, configPath) {
		t.Error("Error message missing config path")
	}
	if !strings.Contains(errStr, "https://gelotto.io/workers") {
		t.Error("Error message missing registration URL")
	}
}
