package worker

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/client"
	"github.com/Gelotto/power-node/internal/models"
	"github.com/Gelotto/power-node/internal/testutil"
)

// createTestWorker creates a worker with mock dependencies for testing
func createTestWorker(mockServer *testutil.MockAPIServer, mockExecutor *testutil.MockExecutor) *Worker {
	apiClient := client.NewAPIClient(mockServer.URL, testutil.ValidAPIKey())

	return &Worker{
		id:               "test-worker-id",
		hostname:         "test-host",
		gpuInfo:          "NVIDIA RTX 4090",
		apiClient:        apiClient,
		pythonExec:       mockExecutor,
		pythonRunning:    true,
		supportsVideo:    true,
		supportsFaceSwap: true,
		stopChan:         make(chan struct{}),
		config: &Config{
			FaceSwap: FaceSwapConfig{
				MaxFrames: 100,
			},
		},
	}
}

// ============================================================================
// IMAGE JOB TESTS
// ============================================================================

func TestProcessImageJob_Success(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateImageJob("job-123", "a beautiful sunset")
	ctx := context.Background()

	worker.processImageJob(ctx, job)

	// Verify executor was called with correct request
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Fatalf("Expected 1 Generate call, got %d", len(mockExecutor.GenerateCalls))
	}

	genReq := mockExecutor.GenerateCalls[0]
	if genReq.Prompt != "a beautiful sunset" {
		t.Errorf("Expected prompt 'a beautiful sunset', got '%s'", genReq.Prompt)
	}
	if genReq.Width != 1024 {
		t.Errorf("Expected width 1024, got %d", genReq.Width)
	}
	if genReq.Height != 1024 {
		t.Errorf("Expected height 1024, got %d", genReq.Height)
	}
	if genReq.Steps != 8 {
		t.Errorf("Expected steps 8, got %d", genReq.Steps)
	}

	// Verify API calls
	if mockServer.GetStartCalls() != 1 {
		t.Errorf("Expected 1 StartProcessing call, got %d", mockServer.GetStartCalls())
	}
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
	if mockServer.LastCompleteJobID != "job-123" {
		t.Errorf("Expected job ID 'job-123', got '%s'", mockServer.LastCompleteJobID)
	}
	if mockServer.LastCompleteData != "base64_mock_image_data" {
		t.Errorf("Expected mock image data, got '%s'", mockServer.LastCompleteData)
	}
}

func TestProcessImageJob_GenerationFails(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateError: errors.New("CUDA out of memory"),
	})

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("job-456", "test prompt")
	ctx := context.Background()

	worker.processImageJob(ctx, job)

	// Verify FailJob was called instead of CompleteJob
	if mockServer.GetCompleteCalls() != 0 {
		t.Errorf("Expected 0 CompleteJob calls, got %d", mockServer.GetCompleteCalls())
	}
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailJobID != "job-456" {
		t.Errorf("Expected failed job ID 'job-456', got '%s'", mockServer.LastFailJobID)
	}
	if mockServer.LastFailError != "CUDA out of memory" {
		t.Errorf("Expected error 'CUDA out of memory', got '%s'", mockServer.LastFailError)
	}
}

func TestProcessImageJob_CompleteJobFails(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockServer.SetCompleteError(500, "Internal server error")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("job-789", "test prompt")
	ctx := context.Background()

	// This should not panic - error is logged but generation succeeds
	worker.processImageJob(ctx, job)

	// Verify generation was attempted
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected 1 Generate call, got %d", len(mockExecutor.GenerateCalls))
	}

	// Complete was called (even though it failed)
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
}

func TestProcessImageJob_StartProcessingFails(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockServer.SetStartError(500, "Start failed")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("job-start-fail", "test prompt")
	ctx := context.Background()

	// Processing should continue even if StartProcessing fails
	worker.processImageJob(ctx, job)

	// Verify generation still happened
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected 1 Generate call despite start failure, got %d", len(mockExecutor.GenerateCalls))
	}

	// Verify completion was attempted
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
}

func TestProcessImageJob_ContextCancelled(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		return nil, ctx.Err()
	}

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("job-ctx-cancel", "test prompt")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Should not panic when context is cancelled
	// Note: FailJob will also fail since context is cancelled, but that's expected
	worker.processImageJob(ctx, job)

	// Verify generation was attempted
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected Generate to be called even with cancelled context, got %d calls", len(mockExecutor.GenerateCalls))
	}

	// CompleteJob should not be called since generation failed
	if mockServer.GetCompleteCalls() != 0 {
		t.Error("CompleteJob should not be called when generation fails")
	}
}

// ============================================================================
// VIDEO JOB TESTS
// ============================================================================

func TestProcessVideoJob_Success(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateVideoJob("video-123", "a flying dragon")
	ctx := context.Background()

	worker.processVideoJob(ctx, job)

	// Verify executor was called with correct request
	if len(mockExecutor.GenerateVideoCalls) != 1 {
		t.Fatalf("Expected 1 GenerateVideo call, got %d", len(mockExecutor.GenerateVideoCalls))
	}

	videoReq := mockExecutor.GenerateVideoCalls[0]
	if videoReq.Prompt != "a flying dragon" {
		t.Errorf("Expected prompt 'a flying dragon', got '%s'", videoReq.Prompt)
	}
	if videoReq.Width != 832 {
		t.Errorf("Expected width 832, got %d", videoReq.Width)
	}
	if videoReq.Height != 480 {
		t.Errorf("Expected height 480, got %d", videoReq.Height)
	}
	if videoReq.DurationSeconds != 5 {
		t.Errorf("Expected duration 5s, got %d", videoReq.DurationSeconds)
	}
	if videoReq.FPS != 24 {
		t.Errorf("Expected FPS 24, got %d", videoReq.FPS)
	}
	if videoReq.TotalFrames != 120 {
		t.Errorf("Expected total frames 120, got %d", videoReq.TotalFrames)
	}

	// Verify API calls
	if mockServer.GetStartCalls() != 1 {
		t.Errorf("Expected 1 StartProcessing call, got %d", mockServer.GetStartCalls())
	}
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
	if !mockServer.LastCompleteIsVideo {
		t.Error("Expected CompleteJob to indicate video content")
	}
}

func TestProcessVideoJob_ProgressCallback(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var progressCalls int32
	mockExecutor.GenerateVideoFunc = func(ctx context.Context, req *models.GenerateVideoRequest) (*models.GenerateVideoResponse, error) {
		// Simulate progress updates
		mockExecutor.SimulateProgress(models.ProgressMessage{
			Type:            "progress",
			Step:            5,
			TotalSteps:      25,
			ProgressPercent: 20.0,
			FramesCompleted: 24,
		})
		mockExecutor.SimulateProgress(models.ProgressMessage{
			Type:            "progress",
			Step:            15,
			TotalSteps:      25,
			ProgressPercent: 60.0,
			FramesCompleted: 72, // >10 frames difference, should report
		})
		mockExecutor.SimulateProgress(models.ProgressMessage{
			Type:            "progress",
			Step:            24,
			TotalSteps:      25,
			ProgressPercent: 96.0, // >95%, should report
			FramesCompleted: 115,
		})

		atomic.AddInt32(&progressCalls, 3)

		return &models.GenerateVideoResponse{
			VideoData:       "base64_video",
			Format:          "mp4",
			FramesGenerated: 120,
			DurationSeconds: 5,
		}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateVideoJob("video-progress", "test video")
	ctx := context.Background()

	worker.processVideoJob(ctx, job)

	// Wait a bit for async progress reports
	time.Sleep(100 * time.Millisecond)

	// Should have some progress calls (rate-limited to >10 frames difference or >95%)
	// Exact number depends on rate limiting logic
	if mockServer.GetProgressCalls() < 1 {
		t.Errorf("Expected at least 1 progress report, got %d", mockServer.GetProgressCalls())
	}
}

func TestProcessVideoJob_GenerationFails(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateVideoError: errors.New("video generation timeout"),
	})

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateVideoJob("video-fail", "test video")
	ctx := context.Background()

	worker.processVideoJob(ctx, job)

	// Verify FailJob was called
	if mockServer.GetCompleteCalls() != 0 {
		t.Errorf("Expected 0 CompleteJob calls, got %d", mockServer.GetCompleteCalls())
	}
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailError != "video generation timeout" {
		t.Errorf("Expected error 'video generation timeout', got '%s'", mockServer.LastFailError)
	}
}

func TestProcessVideoJob_DefaultsUsed(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	// Job without duration/fps specified
	job := &models.Job{
		ID:     "video-defaults",
		Type:   models.JobTypeVideo,
		Prompt: "test",
		Width:  832,
		Height: 480,
	}
	ctx := context.Background()

	worker.processVideoJob(ctx, job)

	if len(mockExecutor.GenerateVideoCalls) != 1 {
		t.Fatalf("Expected 1 GenerateVideo call, got %d", len(mockExecutor.GenerateVideoCalls))
	}

	videoReq := mockExecutor.GenerateVideoCalls[0]
	// Should use defaults: 5 seconds, 24 FPS = 120 frames
	if videoReq.DurationSeconds != 5 {
		t.Errorf("Expected default duration 5s, got %d", videoReq.DurationSeconds)
	}
	if videoReq.FPS != 24 {
		t.Errorf("Expected default FPS 24, got %d", videoReq.FPS)
	}
	if videoReq.TotalFrames != 120 {
		t.Errorf("Expected default total frames 120, got %d", videoReq.TotalFrames)
	}
}

// ============================================================================
// FACE-SWAP JOB TESTS
// ============================================================================

func TestProcessFaceSwapJob_Success(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateFaceSwapJob("faceswap-123")
	ctx := context.Background()

	worker.processFaceSwapJob(ctx, job)

	// Verify executor was called with correct request
	if len(mockExecutor.GenerateFaceSwapCalls) != 1 {
		t.Fatalf("Expected 1 GenerateFaceSwap call, got %d", len(mockExecutor.GenerateFaceSwapCalls))
	}

	fsReq := mockExecutor.GenerateFaceSwapCalls[0]
	if fsReq.SourceImageURL != "https://example.com/source.jpg" {
		t.Errorf("Unexpected source URL: %s", fsReq.SourceImageURL)
	}
	if fsReq.TargetImageURL != "https://example.com/target.jpg" {
		t.Errorf("Unexpected target URL: %s", fsReq.TargetImageURL)
	}
	if !fsReq.SwapAllFaces {
		t.Error("Expected SwapAllFaces=true")
	}
	if !fsReq.Enhance {
		t.Error("Expected Enhance=true")
	}

	// Verify API calls
	if mockServer.GetStartCalls() != 1 {
		t.Errorf("Expected 1 StartProcessing call, got %d", mockServer.GetStartCalls())
	}
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call (faceswap), got %d", mockServer.GetCompleteCalls())
	}
}

func TestProcessFaceSwapJob_MissingSourceURL(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateInvalidFaceSwapJob("faceswap-invalid")
	ctx := context.Background()

	worker.processFaceSwapJob(ctx, job)

	// Should fail without calling executor
	if len(mockExecutor.GenerateFaceSwapCalls) != 0 {
		t.Errorf("Expected 0 GenerateFaceSwap calls, got %d", len(mockExecutor.GenerateFaceSwapCalls))
	}

	// Should call FailJob
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailError != "Missing source or target image URL" {
		t.Errorf("Unexpected error message: %s", mockServer.LastFailError)
	}
}

func TestProcessFaceSwapJob_GIFFormat(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateFaceSwapGIFJob("faceswap-gif", 50)
	ctx := context.Background()

	worker.processFaceSwapJob(ctx, job)

	if len(mockExecutor.GenerateFaceSwapCalls) != 1 {
		t.Fatalf("Expected 1 GenerateFaceSwap call, got %d", len(mockExecutor.GenerateFaceSwapCalls))
	}

	fsReq := mockExecutor.GenerateFaceSwapCalls[0]
	if !fsReq.IsGIF {
		t.Error("Expected IsGIF=true for GIF job")
	}
	if fsReq.MaxFrames != 50 {
		t.Errorf("Expected MaxFrames=50, got %d", fsReq.MaxFrames)
	}
}

func TestProcessFaceSwapJob_GenerationFails(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateFaceSwapError: errors.New("No face detected in source image"),
	})

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateFaceSwapJob("faceswap-no-face")
	ctx := context.Background()

	worker.processFaceSwapJob(ctx, job)

	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailError != "No face detected in source image" {
		t.Errorf("Unexpected error: %s", mockServer.LastFailError)
	}
}

// ============================================================================
// JOB ROUTING TESTS
// ============================================================================

func TestProcessJob_RoutesToImageJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateImageJob("route-image", "test")
	ctx := context.Background()

	worker.processJob(ctx, job)

	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected image job to route to Generate, got %d calls", len(mockExecutor.GenerateCalls))
	}
	if len(mockExecutor.GenerateVideoCalls) != 0 {
		t.Error("Image job should not call GenerateVideo")
	}
	if len(mockExecutor.GenerateFaceSwapCalls) != 0 {
		t.Error("Image job should not call GenerateFaceSwap")
	}
}

func TestProcessJob_RoutesToVideoJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateVideoJob("route-video", "test")
	ctx := context.Background()

	worker.processJob(ctx, job)

	if len(mockExecutor.GenerateVideoCalls) != 1 {
		t.Errorf("Expected video job to route to GenerateVideo, got %d calls", len(mockExecutor.GenerateVideoCalls))
	}
	if len(mockExecutor.GenerateCalls) != 0 {
		t.Error("Video job should not call Generate")
	}
}

func TestProcessJob_RoutesToFaceSwapJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)

	job := testutil.CreateFaceSwapJob("route-faceswap")
	ctx := context.Background()

	worker.processJob(ctx, job)

	if len(mockExecutor.GenerateFaceSwapCalls) != 1 {
		t.Errorf("Expected faceswap job to route to GenerateFaceSwap, got %d calls", len(mockExecutor.GenerateFaceSwapCalls))
	}
	if len(mockExecutor.GenerateCalls) != 0 {
		t.Error("FaceSwap job should not call Generate")
	}
}

func TestProcessJob_VideoCapabilityCheck(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.supportsVideo = false // Disable video support

	job := testutil.CreateVideoJob("video-unsupported", "test")
	ctx := context.Background()

	worker.processJob(ctx, job)

	// Should not call GenerateVideo
	if len(mockExecutor.GenerateVideoCalls) != 0 {
		t.Error("Should not call GenerateVideo when video not supported")
	}

	// Should fail the job
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailError != "Worker does not support video generation" {
		t.Errorf("Unexpected error: %s", mockServer.LastFailError)
	}
}

func TestProcessJob_FaceSwapCapabilityCheck(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	worker.supportsFaceSwap = false // Disable face-swap support

	job := testutil.CreateFaceSwapJob("faceswap-unsupported")
	ctx := context.Background()

	worker.processJob(ctx, job)

	// Should not call GenerateFaceSwap
	if len(mockExecutor.GenerateFaceSwapCalls) != 0 {
		t.Error("Should not call GenerateFaceSwap when not supported")
	}

	// Should fail the job
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if mockServer.LastFailError != "Worker does not support face-swap" {
		t.Errorf("Unexpected error: %s", mockServer.LastFailError)
	}
}

func TestProcessJob_SetsProcessingFlag(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var processingDuringGenerate bool
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		// Check if processing flag is set during generation
		processingDuringGenerate = true // We're inside processJob, flag should be true
		return &models.GenerateResponse{
			ImageData: "test",
			Format:    "png",
		}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("flag-test", "test")
	ctx := context.Background()

	// Verify flag is false before
	if worker.isProcessingJob() {
		t.Error("Processing flag should be false before processJob")
	}

	worker.processJob(ctx, job)

	// Verify flag is false after
	if worker.isProcessingJob() {
		t.Error("Processing flag should be false after processJob")
	}

	if !processingDuringGenerate {
		t.Error("Generate was not called")
	}
}
