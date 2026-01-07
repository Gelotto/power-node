package testutil

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/Gelotto/power-node/internal/models"
)

// MockAPIServer is a configurable mock API server for testing
type MockAPIServer struct {
	*httptest.Server
	mu sync.Mutex

	// Request tracking
	HeartbeatCalls   int32
	ClaimCalls       int32
	StartCalls       int32
	ProgressCalls    int32
	CompleteCalls    int32
	FailCalls        int32

	// Last request data
	LastHeartbeatData map[string]interface{}
	LastCompleteJobID string
	LastCompleteData  string
	LastCompleteIsVideo bool
	LastFailJobID     string
	LastFailError     string
	LastProgressJobID string
	LastProgressFrames int

	// Configurable responses
	ClaimResponses     []*models.Job
	claimIndex         int

	// Failure injection
	HeartbeatError     *HTTPError
	ClaimError         *HTTPError
	StartError         *HTTPError
	ProgressError      *HTTPError
	CompleteError      *HTTPError
	FailError          *HTTPError

	// Custom handlers
	CustomHeartbeatHandler func(w http.ResponseWriter, r *http.Request)
	CustomClaimHandler     func(w http.ResponseWriter, r *http.Request)
	CustomCompleteHandler  func(w http.ResponseWriter, r *http.Request)
	CustomFailHandler      func(w http.ResponseWriter, r *http.Request)
}

// HTTPError represents an HTTP error response
type HTTPError struct {
	StatusCode int
	Message    string
}

// NewMockAPIServer creates a new mock API server
func NewMockAPIServer() *MockAPIServer {
	mock := &MockAPIServer{
		ClaimResponses: make([]*models.Job, 0),
	}

	mock.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mock.handleRequest(w, r)
	}))

	return mock
}

func (m *MockAPIServer) handleRequest(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	switch {
	case strings.HasSuffix(path, "/heartbeat"):
		m.handleHeartbeat(w, r)
	case strings.HasSuffix(path, "/claim"):
		m.handleClaim(w, r)
	case strings.HasSuffix(path, "/processing"):
		m.handleStartProcessing(w, r)
	case strings.HasSuffix(path, "/progress"):
		m.handleProgress(w, r)
	case strings.HasSuffix(path, "/complete"):
		m.handleComplete(w, r)
	case strings.HasSuffix(path, "/fail"):
		m.handleFail(w, r)
	case path == "/health":
		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (m *MockAPIServer) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.HeartbeatCalls, 1)

	if m.CustomHeartbeatHandler != nil {
		m.CustomHeartbeatHandler(w, r)
		return
	}

	if m.HeartbeatError != nil {
		http.Error(w, m.HeartbeatError.Message, m.HeartbeatError.StatusCode)
		return
	}

	// Parse and store heartbeat data
	body, _ := io.ReadAll(r.Body)
	var data map[string]interface{}
	_ = json.Unmarshal(body, &data)

	m.mu.Lock()
	m.LastHeartbeatData = data
	m.mu.Unlock()

	w.WriteHeader(http.StatusOK)
}

func (m *MockAPIServer) handleClaim(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.ClaimCalls, 1)

	if m.CustomClaimHandler != nil {
		m.CustomClaimHandler(w, r)
		return
	}

	if m.ClaimError != nil {
		http.Error(w, m.ClaimError.Message, m.ClaimError.StatusCode)
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.claimIndex < len(m.ClaimResponses) {
		job := m.ClaimResponses[m.claimIndex]
		m.claimIndex++
		if job == nil {
			w.WriteHeader(http.StatusNoContent)
		} else {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(job)
		}
	} else {
		w.WriteHeader(http.StatusNoContent)
	}
}

func (m *MockAPIServer) handleStartProcessing(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.StartCalls, 1)

	if m.StartError != nil {
		http.Error(w, m.StartError.Message, m.StartError.StatusCode)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func (m *MockAPIServer) handleProgress(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.ProgressCalls, 1)

	if m.ProgressError != nil {
		http.Error(w, m.ProgressError.Message, m.ProgressError.StatusCode)
		return
	}

	// Parse progress data
	body, _ := io.ReadAll(r.Body)
	var data map[string]interface{}
	_ = json.Unmarshal(body, &data)

	m.mu.Lock()
	if jobID, ok := data["job_id"].(string); ok {
		m.LastProgressJobID = jobID
	}
	if frames, ok := data["frames_completed"].(float64); ok {
		m.LastProgressFrames = int(frames)
	}
	m.mu.Unlock()

	w.WriteHeader(http.StatusOK)
}

func (m *MockAPIServer) handleComplete(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.CompleteCalls, 1)

	if m.CustomCompleteHandler != nil {
		m.CustomCompleteHandler(w, r)
		return
	}

	if m.CompleteError != nil {
		http.Error(w, m.CompleteError.Message, m.CompleteError.StatusCode)
		return
	}

	// Parse complete data
	body, _ := io.ReadAll(r.Body)
	var data map[string]interface{}
	json.Unmarshal(body, &data)

	m.mu.Lock()
	if jobID, ok := data["job_id"].(string); ok {
		m.LastCompleteJobID = jobID
	}
	if imageData, ok := data["image_data"].(string); ok {
		m.LastCompleteData = imageData
	}
	if videoData, ok := data["video_data"].(string); ok {
		m.LastCompleteData = videoData
		m.LastCompleteIsVideo = true
	}
	m.mu.Unlock()

	w.WriteHeader(http.StatusOK)
}

func (m *MockAPIServer) handleFail(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt32(&m.FailCalls, 1)

	if m.CustomFailHandler != nil {
		m.CustomFailHandler(w, r)
		return
	}

	if m.FailError != nil {
		http.Error(w, m.FailError.Message, m.FailError.StatusCode)
		return
	}

	// Parse fail data
	body, _ := io.ReadAll(r.Body)
	var data map[string]interface{}
	json.Unmarshal(body, &data)

	m.mu.Lock()
	if jobID, ok := data["job_id"].(string); ok {
		m.LastFailJobID = jobID
	}
	if errMsg, ok := data["error"].(string); ok {
		m.LastFailError = errMsg
	}
	m.mu.Unlock()

	w.WriteHeader(http.StatusOK)
}

// QueueJob adds a job to be returned by the next claim
func (m *MockAPIServer) QueueJob(job *models.Job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ClaimResponses = append(m.ClaimResponses, job)
}

// QueueJobs adds multiple jobs to be returned by subsequent claims
func (m *MockAPIServer) QueueJobs(jobs ...*models.Job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ClaimResponses = append(m.ClaimResponses, jobs...)
}

// SetCompleteError configures the complete endpoint to return an error
func (m *MockAPIServer) SetCompleteError(statusCode int, message string) {
	m.CompleteError = &HTTPError{StatusCode: statusCode, Message: message}
}

// SetFailError configures the fail endpoint to return an error
func (m *MockAPIServer) SetFailError(statusCode int, message string) {
	m.FailError = &HTTPError{StatusCode: statusCode, Message: message}
}

// SetHeartbeatError configures the heartbeat endpoint to return an error
func (m *MockAPIServer) SetHeartbeatError(statusCode int, message string) {
	m.HeartbeatError = &HTTPError{StatusCode: statusCode, Message: message}
}

// SetClaimError configures the claim endpoint to return an error
func (m *MockAPIServer) SetClaimError(statusCode int, message string) {
	m.ClaimError = &HTTPError{StatusCode: statusCode, Message: message}
}

// SetStartError configures the start processing endpoint to return an error
func (m *MockAPIServer) SetStartError(statusCode int, message string) {
	m.StartError = &HTTPError{StatusCode: statusCode, Message: message}
}

// SetProgressError configures the progress endpoint to return an error
func (m *MockAPIServer) SetProgressError(statusCode int, message string) {
	m.ProgressError = &HTTPError{StatusCode: statusCode, Message: message}
}

// Reset clears all state and errors
func (m *MockAPIServer) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	atomic.StoreInt32(&m.HeartbeatCalls, 0)
	atomic.StoreInt32(&m.ClaimCalls, 0)
	atomic.StoreInt32(&m.StartCalls, 0)
	atomic.StoreInt32(&m.ProgressCalls, 0)
	atomic.StoreInt32(&m.CompleteCalls, 0)
	atomic.StoreInt32(&m.FailCalls, 0)

	m.LastHeartbeatData = nil
	m.LastCompleteJobID = ""
	m.LastCompleteData = ""
	m.LastCompleteIsVideo = false
	m.LastFailJobID = ""
	m.LastFailError = ""
	m.LastProgressJobID = ""
	m.LastProgressFrames = 0

	m.ClaimResponses = make([]*models.Job, 0)
	m.claimIndex = 0

	m.HeartbeatError = nil
	m.ClaimError = nil
	m.StartError = nil
	m.ProgressError = nil
	m.CompleteError = nil
	m.FailError = nil

	m.CustomHeartbeatHandler = nil
	m.CustomClaimHandler = nil
	m.CustomCompleteHandler = nil
	m.CustomFailHandler = nil
}

// GetHeartbeatCalls returns the number of heartbeat calls (thread-safe)
func (m *MockAPIServer) GetHeartbeatCalls() int {
	return int(atomic.LoadInt32(&m.HeartbeatCalls))
}

// GetClaimCalls returns the number of claim calls (thread-safe)
func (m *MockAPIServer) GetClaimCalls() int {
	return int(atomic.LoadInt32(&m.ClaimCalls))
}

// GetStartCalls returns the number of start processing calls (thread-safe)
func (m *MockAPIServer) GetStartCalls() int {
	return int(atomic.LoadInt32(&m.StartCalls))
}

// GetProgressCalls returns the number of progress calls (thread-safe)
func (m *MockAPIServer) GetProgressCalls() int {
	return int(atomic.LoadInt32(&m.ProgressCalls))
}

// GetCompleteCalls returns the number of complete calls (thread-safe)
func (m *MockAPIServer) GetCompleteCalls() int {
	return int(atomic.LoadInt32(&m.CompleteCalls))
}

// GetFailCalls returns the number of fail calls (thread-safe)
func (m *MockAPIServer) GetFailCalls() int {
	return int(atomic.LoadInt32(&m.FailCalls))
}
