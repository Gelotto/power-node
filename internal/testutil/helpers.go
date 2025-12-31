package testutil

import (
	"strings"
	"testing"
	"time"

	"github.com/Gelotto/power-node/internal/models"
)

// ValidAPIKey returns a valid API key for testing
func ValidAPIKey() string {
	return "wk_" + strings.Repeat("a", 64)
}

// CreateImageJob creates a test image job
func CreateImageJob(id, prompt string) *models.Job {
	return &models.Job{
		ID:     id,
		Type:   models.JobTypeImage,
		Prompt: prompt,
		Width:  1024,
		Height: 1024,
		Steps:  8,
	}
}

// CreateVideoJob creates a test video job
func CreateVideoJob(id, prompt string) *models.Job {
	duration := 5
	fps := 24
	totalFrames := 120
	return &models.Job{
		ID:              id,
		Type:            models.JobTypeVideo,
		Prompt:          prompt,
		Width:           832,
		Height:          480,
		Steps:           25,
		DurationSeconds: &duration,
		FPS:             &fps,
		TotalFrames:     &totalFrames,
	}
}

// CreateFaceSwapJob creates a test face-swap job
func CreateFaceSwapJob(id string) *models.Job {
	sourceURL := "https://example.com/source.jpg"
	targetURL := "https://example.com/target.jpg"
	swapAll := true
	enhance := true
	return &models.Job{
		ID:             id,
		Type:           models.JobTypeFaceSwap,
		SourceImageURL: &sourceURL,
		TargetImageURL: &targetURL,
		SwapAllFaces:   &swapAll,
		EnhanceResult:  &enhance,
	}
}

// CreateFaceSwapGIFJob creates a test face-swap GIF job
func CreateFaceSwapGIFJob(id string, frameCount int) *models.Job {
	job := CreateFaceSwapJob(id)
	isGIF := true
	job.IsGIF = &isGIF
	job.GifFrameCount = &frameCount
	return job
}

// CreateInvalidFaceSwapJob creates a face-swap job with missing URLs
func CreateInvalidFaceSwapJob(id string) *models.Job {
	return &models.Job{
		ID:   id,
		Type: models.JobTypeFaceSwap,
		// Missing SourceImageURL and TargetImageURL
	}
}

// WaitForCondition waits for a condition to be true or times out
func WaitForCondition(t *testing.T, condition func() bool, timeout time.Duration, message string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("Timeout waiting for condition: %s", message)
}

// WaitForCallCount waits for a call count to reach expected value
func WaitForCallCount(t *testing.T, getCalls func() int, expected int, timeout time.Duration, name string) {
	t.Helper()
	WaitForCondition(t, func() bool {
		return getCalls() >= expected
	}, timeout, name+" call count")
}

// AssertNoError fails the test if err is not nil
func AssertNoError(t *testing.T, err error, msg string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", msg, err)
	}
}

// AssertError fails the test if err is nil
func AssertError(t *testing.T, err error, msg string) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: expected error but got nil", msg)
	}
}

// AssertErrorContains fails if err is nil or doesn't contain expected substring
func AssertErrorContains(t *testing.T, err error, expected string, msg string) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: expected error containing %q but got nil", msg, expected)
	}
	if !strings.Contains(err.Error(), expected) {
		t.Fatalf("%s: expected error containing %q but got %q", msg, expected, err.Error())
	}
}

// AssertEqual fails if got != want
func AssertEqual[T comparable](t *testing.T, got, want T, msg string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v, want %v", msg, got, want)
	}
}

// AssertTrue fails if condition is false
func AssertTrue(t *testing.T, condition bool, msg string) {
	t.Helper()
	if !condition {
		t.Fatalf("%s: expected true but got false", msg)
	}
}

// AssertFalse fails if condition is true
func AssertFalse(t *testing.T, condition bool, msg string) {
	t.Helper()
	if condition {
		t.Fatalf("%s: expected false but got true", msg)
	}
}

// Int64Ptr returns a pointer to an int64
func Int64Ptr(v int64) *int64 {
	return &v
}

// IntPtr returns a pointer to an int
func IntPtr(v int) *int {
	return &v
}

// BoolPtr returns a pointer to a bool
func BoolPtr(v bool) *bool {
	return &v
}

// StringPtr returns a pointer to a string
func StringPtr(v string) *string {
	return &v
}
