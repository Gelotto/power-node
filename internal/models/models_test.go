package models

import (
	"encoding/json"
	"testing"
)

func TestJobType_Constants(t *testing.T) {
	tests := []struct {
		name     string
		jobType  JobType
		expected string
	}{
		{"image type", JobTypeImage, "image"},
		{"video type", JobTypeVideo, "video"},
		{"face_swap type", JobTypeFaceSwap, "face_swap"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.jobType) != tt.expected {
				t.Errorf("JobType constant = %q, want %q", tt.jobType, tt.expected)
			}
		})
	}
}

func TestJob_IsVideoJob(t *testing.T) {
	tests := []struct {
		name     string
		jobType  JobType
		expected bool
	}{
		{"image job returns false", JobTypeImage, false},
		{"video job returns true", JobTypeVideo, true},
		{"face_swap job returns false", JobTypeFaceSwap, false},
		{"empty type returns false", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			job := &Job{Type: tt.jobType}
			if got := job.IsVideoJob(); got != tt.expected {
				t.Errorf("IsVideoJob() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestJob_IsFaceSwapJob(t *testing.T) {
	tests := []struct {
		name     string
		jobType  JobType
		expected bool
	}{
		{"image job returns false", JobTypeImage, false},
		{"video job returns false", JobTypeVideo, false},
		{"face_swap job returns true", JobTypeFaceSwap, true},
		{"empty type returns false", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			job := &Job{Type: tt.jobType}
			if got := job.IsFaceSwapJob(); got != tt.expected {
				t.Errorf("IsFaceSwapJob() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestJob_JSONSerialization(t *testing.T) {
	seed := int64(12345)
	negPrompt := "blurry, low quality"
	duration := 5
	fps := 24
	frames := 120

	tests := []struct {
		name string
		job  Job
	}{
		{
			name: "image job",
			job: Job{
				ID:             "test-id-123",
				Type:           JobTypeImage,
				Status:         "pending",
				Prompt:         "a beautiful sunset",
				NegativePrompt: &negPrompt,
				Width:          1024,
				Height:         1024,
				Steps:          8,
				Seed:           &seed,
			},
		},
		{
			name: "video job",
			job: Job{
				ID:              "video-id-456",
				Type:            JobTypeVideo,
				Status:          "claimed",
				Prompt:          "cat dancing",
				Width:           480,
				Height:          480,
				Steps:           25,
				DurationSeconds: &duration,
				FPS:             &fps,
				TotalFrames:     &frames,
			},
		},
		{
			name: "face_swap job",
			job: Job{
				ID:     "swap-id-789",
				Type:   JobTypeFaceSwap,
				Status: "processing",
				Prompt: "",
				Width:  512,
				Height: 512,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal
			data, err := json.Marshal(tt.job)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}

			// Unmarshal
			var decoded Job
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}

			// Verify fields
			if decoded.ID != tt.job.ID {
				t.Errorf("ID = %q, want %q", decoded.ID, tt.job.ID)
			}
			if decoded.Type != tt.job.Type {
				t.Errorf("Type = %q, want %q", decoded.Type, tt.job.Type)
			}
			if decoded.Status != tt.job.Status {
				t.Errorf("Status = %q, want %q", decoded.Status, tt.job.Status)
			}
			if decoded.Width != tt.job.Width {
				t.Errorf("Width = %d, want %d", decoded.Width, tt.job.Width)
			}
			if decoded.Height != tt.job.Height {
				t.Errorf("Height = %d, want %d", decoded.Height, tt.job.Height)
			}
		})
	}
}

func TestJob_JSONDeserialization_FromBackend(t *testing.T) {
	// Test deserializing JSON that looks like what the backend sends
	tests := []struct {
		name        string
		jsonInput   string
		wantType    JobType
		wantIsVideo bool
		wantIsFace  bool
	}{
		{
			name:        "image job from backend",
			jsonInput:   `{"id":"abc123","type":"image","status":"pending","prompt":"test","width":1024,"height":1024,"steps":8}`,
			wantType:    JobTypeImage,
			wantIsVideo: false,
			wantIsFace:  false,
		},
		{
			name:        "video job from backend",
			jsonInput:   `{"id":"def456","type":"video","status":"claimed","prompt":"video test","width":480,"height":480,"steps":25,"duration_seconds":5,"fps":24,"total_frames":120}`,
			wantType:    JobTypeVideo,
			wantIsVideo: true,
			wantIsFace:  false,
		},
		{
			name:        "face_swap job from backend",
			jsonInput:   `{"id":"ghi789","type":"face_swap","status":"processing","prompt":"","width":512,"height":512,"steps":0,"source_image_url":"https://example.com/source.jpg","target_image_url":"https://example.com/target.jpg","is_gif":false,"swap_all_faces":true,"enhance_result":true}`,
			wantType:    JobTypeFaceSwap,
			wantIsVideo: false,
			wantIsFace:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var job Job
			if err := json.Unmarshal([]byte(tt.jsonInput), &job); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}

			if job.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", job.Type, tt.wantType)
			}
			if job.IsVideoJob() != tt.wantIsVideo {
				t.Errorf("IsVideoJob() = %v, want %v", job.IsVideoJob(), tt.wantIsVideo)
			}
			if job.IsFaceSwapJob() != tt.wantIsFace {
				t.Errorf("IsFaceSwapJob() = %v, want %v", job.IsFaceSwapJob(), tt.wantIsFace)
			}
		})
	}
}

func TestJob_OptionalFields(t *testing.T) {
	// Test that optional fields are properly nil when not present
	jsonInput := `{"id":"test","type":"image","status":"pending","prompt":"test","width":1024,"height":1024,"steps":8}`

	var job Job
	if err := json.Unmarshal([]byte(jsonInput), &job); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if job.NegativePrompt != nil {
		t.Error("NegativePrompt should be nil when not present")
	}
	if job.Seed != nil {
		t.Error("Seed should be nil when not present")
	}
	if job.DurationSeconds != nil {
		t.Error("DurationSeconds should be nil when not present")
	}
	if job.FPS != nil {
		t.Error("FPS should be nil when not present")
	}
	if job.SourceImageURL != nil {
		t.Error("SourceImageURL should be nil when not present")
	}
}

func TestGenerateRequest_JSONSerialization(t *testing.T) {
	seed := int64(42)
	negPrompt := "ugly"

	req := GenerateRequest{
		Prompt:         "a cat",
		NegativePrompt: &negPrompt,
		Width:          512,
		Height:         512,
		Steps:          8,
		Seed:           &seed,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded GenerateRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.Prompt != req.Prompt {
		t.Errorf("Prompt = %q, want %q", decoded.Prompt, req.Prompt)
	}
	if decoded.Width != req.Width {
		t.Errorf("Width = %d, want %d", decoded.Width, req.Width)
	}
	if *decoded.Seed != *req.Seed {
		t.Errorf("Seed = %d, want %d", *decoded.Seed, *req.Seed)
	}
}

func TestGenerateResponse_JSONSerialization(t *testing.T) {
	resp := GenerateResponse{
		ImageData: "base64encodeddata==",
		Format:    "png",
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded GenerateResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.ImageData != resp.ImageData {
		t.Errorf("ImageData = %q, want %q", decoded.ImageData, resp.ImageData)
	}
	if decoded.Format != resp.Format {
		t.Errorf("Format = %q, want %q", decoded.Format, resp.Format)
	}
}

func TestJSONRPCRequest_Serialization(t *testing.T) {
	req := JSONRPCRequest{
		ID:     1,
		Method: "generate",
		Params: GenerateRequest{
			Prompt: "test",
			Width:  1024,
			Height: 1024,
			Steps:  8,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// Verify the JSON structure
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if raw["id"].(float64) != 1 {
		t.Errorf("id = %v, want 1", raw["id"])
	}
	if raw["method"].(string) != "generate" {
		t.Errorf("method = %v, want generate", raw["method"])
	}
	if _, ok := raw["params"]; !ok {
		t.Error("params field missing")
	}
}

func TestJSONRPCResponse_WithResult(t *testing.T) {
	jsonInput := `{"id":1,"result":{"image_data":"abc123==","format":"png"},"error":null}`

	var resp JSONRPCResponse
	if err := json.Unmarshal([]byte(jsonInput), &resp); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if resp.ID != 1 {
		t.Errorf("ID = %d, want 1", resp.ID)
	}
	if resp.Result == nil {
		t.Fatal("Result should not be nil")
	}
	if resp.Result.ImageData != "abc123==" {
		t.Errorf("ImageData = %q, want %q", resp.Result.ImageData, "abc123==")
	}
	if resp.Error != nil {
		t.Error("Error should be nil")
	}
}

func TestJSONRPCResponse_WithError(t *testing.T) {
	errMsg := "CUDA out of memory"
	jsonInput := `{"id":2,"result":null,"error":"CUDA out of memory"}`

	var resp JSONRPCResponse
	if err := json.Unmarshal([]byte(jsonInput), &resp); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if resp.ID != 2 {
		t.Errorf("ID = %d, want 2", resp.ID)
	}
	if resp.Result != nil {
		t.Error("Result should be nil on error")
	}
	if resp.Error == nil {
		t.Fatal("Error should not be nil")
	}
	if *resp.Error != errMsg {
		t.Errorf("Error = %q, want %q", *resp.Error, errMsg)
	}
}

func TestProgressMessage_Serialization(t *testing.T) {
	msg := ProgressMessage{
		Type:            "progress",
		Step:            10,
		TotalSteps:      25,
		ProgressPercent: 40.0,
		FramesCompleted: 48,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded ProgressMessage
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.Type != "progress" {
		t.Errorf("Type = %q, want progress", decoded.Type)
	}
	if decoded.Step != 10 {
		t.Errorf("Step = %d, want 10", decoded.Step)
	}
	if decoded.TotalSteps != 25 {
		t.Errorf("TotalSteps = %d, want 25", decoded.TotalSteps)
	}
	if decoded.ProgressPercent != 40.0 {
		t.Errorf("ProgressPercent = %f, want 40.0", decoded.ProgressPercent)
	}
}

func TestFaceSwapRequest_Serialization(t *testing.T) {
	req := FaceSwapRequest{
		SourceImageURL: "https://example.com/source.jpg",
		TargetImageURL: "https://example.com/target.jpg",
		IsGIF:          true,
		SwapAllFaces:   true,
		Enhance:        true,
		MaxFrames:      50,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded FaceSwapRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.SourceImageURL != req.SourceImageURL {
		t.Errorf("SourceImageURL = %q, want %q", decoded.SourceImageURL, req.SourceImageURL)
	}
	if decoded.IsGIF != req.IsGIF {
		t.Errorf("IsGIF = %v, want %v", decoded.IsGIF, req.IsGIF)
	}
	if decoded.MaxFrames != req.MaxFrames {
		t.Errorf("MaxFrames = %d, want %d", decoded.MaxFrames, req.MaxFrames)
	}
}

func TestGenerateVideoRequest_Serialization(t *testing.T) {
	seed := int64(99999)
	steps := 35
	guidance := 5.5

	req := GenerateVideoRequest{
		Prompt:          "dancing cat",
		Width:           480,
		Height:          480,
		DurationSeconds: 5,
		FPS:             24,
		TotalFrames:     120,
		Seed:            &seed,
		Steps:           &steps,
		GuidanceScale:   &guidance,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded GenerateVideoRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.Prompt != req.Prompt {
		t.Errorf("Prompt = %q, want %q", decoded.Prompt, req.Prompt)
	}
	if decoded.DurationSeconds != req.DurationSeconds {
		t.Errorf("DurationSeconds = %d, want %d", decoded.DurationSeconds, req.DurationSeconds)
	}
	if decoded.FPS != req.FPS {
		t.Errorf("FPS = %d, want %d", decoded.FPS, req.FPS)
	}
	if *decoded.Steps != *req.Steps {
		t.Errorf("Steps = %d, want %d", *decoded.Steps, *req.Steps)
	}
	if *decoded.GuidanceScale != *req.GuidanceScale {
		t.Errorf("GuidanceScale = %f, want %f", *decoded.GuidanceScale, *req.GuidanceScale)
	}
}

func TestGenerateVideoResponse_Serialization(t *testing.T) {
	resp := GenerateVideoResponse{
		VideoData:       "base64videodatahere==",
		Format:          "mp4",
		FramesGenerated: 120,
		DurationSeconds: 5,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded GenerateVideoResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.VideoData != resp.VideoData {
		t.Errorf("VideoData = %q, want %q", decoded.VideoData, resp.VideoData)
	}
	if decoded.Format != resp.Format {
		t.Errorf("Format = %q, want %q", decoded.Format, resp.Format)
	}
	if decoded.FramesGenerated != resp.FramesGenerated {
		t.Errorf("FramesGenerated = %d, want %d", decoded.FramesGenerated, resp.FramesGenerated)
	}
}

func TestWorkerInfo_Serialization(t *testing.T) {
	info := WorkerInfo{
		ID:            "worker-123",
		Status:        "online",
		Hostname:      "gpu-server-1",
		ModelVersion:  "z-image-turbo",
		JobsCompleted: 100,
	}

	data, err := json.Marshal(info)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var decoded WorkerInfo
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if decoded.ID != info.ID {
		t.Errorf("ID = %q, want %q", decoded.ID, info.ID)
	}
	if decoded.JobsCompleted != info.JobsCompleted {
		t.Errorf("JobsCompleted = %d, want %d", decoded.JobsCompleted, info.JobsCompleted)
	}
}
