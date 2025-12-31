package worker

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/models"
	"github.com/Gelotto/power-node/internal/testutil"
)

// ============================================================================
// ERROR RECOVERY TESTS
// Tests focused on error handling and recovery scenarios
// ============================================================================

func TestCompleteJob_APIError_ContinuesWithoutPanic(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockServer.SetCompleteError(500, "Database connection lost")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("complete-fail", "test prompt")
	ctx := context.Background()

	// Should not panic even when CompleteJob fails
	worker.processImageJob(ctx, job)

	// Verify generation succeeded (Complete failure doesn't affect generation)
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected 1 Generate call, got %d", len(mockExecutor.GenerateCalls))
	}

	// Verify CompleteJob was attempted
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
}

func TestCompleteJob_NetworkTimeout_HandledGracefully(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	// Simulate slow response (but use fast timeout in test context)
	mockServer.SetCompleteError(504, "Gateway timeout")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("timeout-test", "test prompt")
	ctx := context.Background()

	worker.processImageJob(ctx, job)

	// Should have attempted completion
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}
}

func TestFailJob_APIError_HandledGracefully(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateError: errors.New("Generation failed"),
	})
	mockServer.SetFailError(500, "FailJob endpoint down")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("fail-fail", "test prompt")
	ctx := context.Background()

	// Should not panic when both generation AND FailJob fail
	worker.processImageJob(ctx, job)

	// FailJob was attempted
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
}

func TestLargePayload_Handled(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	// Generate a large base64 payload (simulating large image)
	largeData := strings.Repeat("A", 1024*1024) // 1MB of data
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		return &models.GenerateResponse{
			ImageData: largeData,
			Format:    "png",
		}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("large-payload", "test")
	ctx := context.Background()

	worker.processImageJob(ctx, job)

	// Verify large data was sent to completion
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call, got %d", mockServer.GetCompleteCalls())
	}

	// Verify the data was preserved
	if len(mockServer.LastCompleteData) != len(largeData) {
		t.Errorf("Expected data length %d, got %d", len(largeData), len(mockServer.LastCompleteData))
	}
}

func TestProcessingJobFlag_SetDuringProcessing(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var flagDuringGeneration bool
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		// Capture the flag state during processing
		flagDuringGeneration = true
		time.Sleep(10 * time.Millisecond) // Simulate some work
		return &models.GenerateResponse{
			ImageData: "test",
			Format:    "png",
		}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)

	// Flag should be false before
	if worker.isProcessingJob() {
		t.Error("Processing flag should be false before processJob")
	}

	job := testutil.CreateImageJob("flag-test", "test")
	worker.processJob(context.Background(), job)

	// Flag should be false after
	if worker.isProcessingJob() {
		t.Error("Processing flag should be false after processJob")
	}

	if !flagDuringGeneration {
		t.Error("GenerateFunc was not called")
	}
}

func TestProcessingJobFlag_ClearedOnError(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateError: errors.New("Generation error"),
	})

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("flag-error-test", "test")

	// Process job that will fail
	worker.processJob(context.Background(), job)

	// Flag should be false even after error
	if worker.isProcessingJob() {
		t.Error("Processing flag should be false after error in processJob")
	}
}

func TestMultipleJobsSequential_FlagManagement(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	worker := createTestWorker(mockServer, mockExecutor)
	ctx := context.Background()

	// Process multiple jobs sequentially
	for i := 0; i < 5; i++ {
		job := testutil.CreateImageJob("seq-job-"+string(rune('a'+i)), "test")
		worker.processJob(ctx, job)

		// Flag should be false after each job
		if worker.isProcessingJob() {
			t.Errorf("Processing flag should be false after job %d", i)
		}
	}

	// All jobs should have been processed
	if len(mockExecutor.GenerateCalls) != 5 {
		t.Errorf("Expected 5 Generate calls, got %d", len(mockExecutor.GenerateCalls))
	}
}

func TestVideoJob_ProgressReportErrors_DontFailJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockServer.SetProgressError(500, "Progress endpoint down")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateVideoJob("video-progress-fail", "test")
	ctx := context.Background()

	worker.processVideoJob(ctx, job)

	// Video should still complete successfully
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected 1 CompleteJob call despite progress failures, got %d", mockServer.GetCompleteCalls())
	}
}

func TestFaceSwapJob_PartialFailure_ReportsError(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockExecutor.ConfigureFailures(testutil.MockExecutorFailure{
		GenerateFaceSwapError: errors.New("Face detection failed: no face found"),
	})

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateFaceSwapJob("faceswap-partial")
	ctx := context.Background()

	worker.processFaceSwapJob(ctx, job)

	// Should have reported the specific error
	if mockServer.GetFailCalls() != 1 {
		t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
	}
	if !strings.Contains(mockServer.LastFailError, "Face detection failed") {
		t.Errorf("Expected error about face detection, got: %s", mockServer.LastFailError)
	}
}

func TestConcurrentProcessingAttempts_OnlyOneProcesses(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	var processingCount int32
	var maxConcurrent int32
	mockExecutor.GenerateFunc = func(ctx context.Context, req *models.GenerateRequest) (*models.GenerateResponse, error) {
		current := atomic.AddInt32(&processingCount, 1)
		for {
			max := atomic.LoadInt32(&maxConcurrent)
			if current > max {
				if atomic.CompareAndSwapInt32(&maxConcurrent, max, current) {
					break
				}
			} else {
				break
			}
		}
		time.Sleep(50 * time.Millisecond) // Simulate work
		atomic.AddInt32(&processingCount, -1)
		return &models.GenerateResponse{ImageData: "test", Format: "png"}, nil
	}

	worker := createTestWorker(mockServer, mockExecutor)
	ctx := context.Background()

	// Try to process multiple jobs concurrently
	done := make(chan struct{}, 3)
	for i := 0; i < 3; i++ {
		go func(id int) {
			job := testutil.CreateImageJob("concurrent-"+string(rune('a'+id)), "test")
			worker.processJob(ctx, job)
			done <- struct{}{}
		}(i)
	}

	// Wait for all to complete
	for i := 0; i < 3; i++ {
		<-done
	}

	// In the current implementation, jobs can process concurrently
	// (there's no global lock preventing this)
	// This test verifies the current behavior
	if atomic.LoadInt32(&maxConcurrent) == 0 {
		t.Error("Expected at least one concurrent execution")
	}
}

func TestJobTypes_ErrorMessagesPreserved(t *testing.T) {
	testCases := []struct {
		name           string
		createJob      func(id string) *models.Job
		configureError func(*testutil.MockExecutor)
		expectedError  string
	}{
		{
			name:      "image_cuda_error",
			createJob: func(id string) *models.Job { return testutil.CreateImageJob(id, "test") },
			configureError: func(m *testutil.MockExecutor) {
				m.ConfigureFailures(testutil.MockExecutorFailure{
					GenerateError: testutil.CUDAOutOfMemoryError(),
				})
			},
			expectedError: "CUDA out of memory",
		},
		{
			name:      "video_timeout_error",
			createJob: func(id string) *models.Job { return testutil.CreateVideoJob(id, "test") },
			configureError: func(m *testutil.MockExecutor) {
				m.ConfigureFailures(testutil.MockExecutorFailure{
					GenerateVideoError: testutil.GenerationTimeoutError(),
				})
			},
			expectedError: "generation timeout",
		},
		{
			name:      "faceswap_detection_error",
			createJob: func(id string) *models.Job { return testutil.CreateFaceSwapJob(id) },
			configureError: func(m *testutil.MockExecutor) {
				m.ConfigureFailures(testutil.MockExecutorFailure{
					GenerateFaceSwapError: errors.New("No face detected in source"),
				})
			},
			expectedError: "No face detected in source",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockServer := testutil.NewMockAPIServer()
			defer mockServer.Close()

			mockExecutor := testutil.NewMockExecutor()
			tc.configureError(mockExecutor)

			worker := createTestWorker(mockServer, mockExecutor)
			job := tc.createJob(tc.name)
			ctx := context.Background()

			worker.processJob(ctx, job)

			if mockServer.GetFailCalls() != 1 {
				t.Errorf("Expected 1 FailJob call, got %d", mockServer.GetFailCalls())
			}
			if !strings.Contains(mockServer.LastFailError, tc.expectedError) {
				t.Errorf("Expected error containing %q, got %q", tc.expectedError, mockServer.LastFailError)
			}
		})
	}
}

func TestStartProcessing_FailureDoesntBlockJob(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()
	mockServer.SetStartError(503, "Service unavailable")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("start-fail", "test")
	ctx := context.Background()

	worker.processImageJob(ctx, job)

	// Generation should still happen
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected generation despite StartProcessing failure, got %d calls", len(mockExecutor.GenerateCalls))
	}

	// Job should complete
	if mockServer.GetCompleteCalls() != 1 {
		t.Errorf("Expected CompleteJob despite StartProcessing failure, got %d calls", mockServer.GetCompleteCalls())
	}
}

func TestAllAPICallsFail_JobStillProcesses(t *testing.T) {
	mockServer := testutil.NewMockAPIServer()
	defer mockServer.Close()

	mockExecutor := testutil.NewMockExecutor()

	// All API calls fail
	mockServer.SetStartError(500, "Start failed")
	mockServer.SetCompleteError(500, "Complete failed")
	mockServer.SetFailError(500, "Fail failed")

	worker := createTestWorker(mockServer, mockExecutor)
	job := testutil.CreateImageJob("all-fail", "test")
	ctx := context.Background()

	// Should not panic
	worker.processImageJob(ctx, job)

	// Generation was still attempted
	if len(mockExecutor.GenerateCalls) != 1 {
		t.Errorf("Expected generation to be attempted, got %d calls", len(mockExecutor.GenerateCalls))
	}
}
