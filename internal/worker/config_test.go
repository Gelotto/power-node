package worker

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadConfig_Defaults(t *testing.T) {
	// Create a minimal config file
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker-id"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	// Check defaults are applied
	if cfg.API.URL != "https://api.gelotto.io" {
		t.Errorf("API.URL = %q, want %q", cfg.API.URL, "https://api.gelotto.io")
	}
	if cfg.Model.Name != "z-image-turbo" {
		t.Errorf("Model.Name = %q, want %q", cfg.Model.Name, "z-image-turbo")
	}
	if cfg.Worker.PollInterval != 5*time.Second {
		t.Errorf("Worker.PollInterval = %v, want %v", cfg.Worker.PollInterval, 5*time.Second)
	}
	if cfg.Worker.HeartbeatInterval != 30*time.Second {
		t.Errorf("Worker.HeartbeatInterval = %v, want %v", cfg.Worker.HeartbeatInterval, 30*time.Second)
	}
	if cfg.Python.Executable != "python3" {
		t.Errorf("Python.Executable = %q, want %q", cfg.Python.Executable, "python3")
	}
	if cfg.Python.ScriptPath != "scripts/inference.py" {
		t.Errorf("Python.ScriptPath = %q, want %q", cfg.Python.ScriptPath, "scripts/inference.py")
	}
}

func TestLoadConfig_VideoDefaults(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Video.MaxDuration != 5 {
		t.Errorf("Video.MaxDuration = %d, want %d", cfg.Video.MaxDuration, 5)
	}
	if cfg.Video.MaxFPS != 24 {
		t.Errorf("Video.MaxFPS = %d, want %d", cfg.Video.MaxFPS, 24)
	}
	if cfg.Video.MaxWidth != 832 {
		t.Errorf("Video.MaxWidth = %d, want %d", cfg.Video.MaxWidth, 832)
	}
	if cfg.Video.MaxHeight != 480 {
		t.Errorf("Video.MaxHeight = %d, want %d", cfg.Video.MaxHeight, 480)
	}
}

func TestLoadConfig_FaceSwapDefaults(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.FaceSwap.MaxFrames != 100 {
		t.Errorf("FaceSwap.MaxFrames = %d, want %d", cfg.FaceSwap.MaxFrames, 100)
	}
	if cfg.FaceSwap.MaxWidth != 2048 {
		t.Errorf("FaceSwap.MaxWidth = %d, want %d", cfg.FaceSwap.MaxWidth, 2048)
	}
	if cfg.FaceSwap.MaxHeight != 2048 {
		t.Errorf("FaceSwap.MaxHeight = %d, want %d", cfg.FaceSwap.MaxHeight, 2048)
	}
	if cfg.FaceSwap.Enhancement == nil || *cfg.FaceSwap.Enhancement != true {
		t.Error("FaceSwap.Enhancement should default to true")
	}
}

func TestLoadConfig_IdleDetectionDefaults(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	idle := cfg.Worker.IdleDetection
	if idle.Enabled != false {
		t.Error("IdleDetection.Enabled should default to false (opt-in)")
	}
	if idle.UtilizationThreshold != 50 {
		t.Errorf("IdleDetection.UtilizationThreshold = %d, want %d", idle.UtilizationThreshold, 50)
	}
	if idle.MemoryThreshold != 80 {
		t.Errorf("IdleDetection.MemoryThreshold = %d, want %d", idle.MemoryThreshold, 80)
	}
	if idle.CheckInterval != 5*time.Second {
		t.Errorf("IdleDetection.CheckInterval = %v, want %v", idle.CheckInterval, 5*time.Second)
	}
	if idle.CooldownPeriod != 30*time.Second {
		t.Errorf("IdleDetection.CooldownPeriod = %v, want %v", idle.CooldownPeriod, 30*time.Second)
	}
}

func TestLoadConfig_CustomValues(t *testing.T) {
	content := `
api:
  url: "https://custom.api.com"
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
model:
  path: "/custom/model/path"
  name: "custom-model"
  service_mode: "pytorch"
  vram_gb: 24
worker:
  id: "custom-worker-id"
  hostname: "custom-host"
  poll_interval: 10s
  heartbeat_interval: 60s
python:
  executable: "/usr/bin/python3.11"
  script_path: "/custom/script.py"
  script_args:
    - "--arg1"
    - "--arg2"
  env:
    CUSTOM_VAR: "value"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.API.URL != "https://custom.api.com" {
		t.Errorf("API.URL = %q, want %q", cfg.API.URL, "https://custom.api.com")
	}
	if cfg.Model.Path != "/custom/model/path" {
		t.Errorf("Model.Path = %q, want %q", cfg.Model.Path, "/custom/model/path")
	}
	if cfg.Model.Name != "custom-model" {
		t.Errorf("Model.Name = %q, want %q", cfg.Model.Name, "custom-model")
	}
	if cfg.Model.ServiceMode != "pytorch" {
		t.Errorf("Model.ServiceMode = %q, want %q", cfg.Model.ServiceMode, "pytorch")
	}
	if cfg.Model.VRAMGB != 24 {
		t.Errorf("Model.VRAMGB = %d, want %d", cfg.Model.VRAMGB, 24)
	}
	if cfg.Worker.PollInterval != 10*time.Second {
		t.Errorf("Worker.PollInterval = %v, want %v", cfg.Worker.PollInterval, 10*time.Second)
	}
	if cfg.Worker.HeartbeatInterval != 60*time.Second {
		t.Errorf("Worker.HeartbeatInterval = %v, want %v", cfg.Worker.HeartbeatInterval, 60*time.Second)
	}
	if cfg.Python.Executable != "/usr/bin/python3.11" {
		t.Errorf("Python.Executable = %q, want %q", cfg.Python.Executable, "/usr/bin/python3.11")
	}
	if len(cfg.Python.ScriptArgs) != 2 {
		t.Errorf("Python.ScriptArgs length = %d, want %d", len(cfg.Python.ScriptArgs), 2)
	}
	if cfg.Python.Env["CUSTOM_VAR"] != "value" {
		t.Errorf("Python.Env[CUSTOM_VAR] = %q, want %q", cfg.Python.Env["CUSTOM_VAR"], "value")
	}
}

func TestLoadConfig_FileNotFound(t *testing.T) {
	_, err := LoadConfig("/nonexistent/path/config.yaml")
	if err == nil {
		t.Error("LoadConfig() should return error for nonexistent file")
	}
}

func TestLoadConfig_InvalidYAML(t *testing.T) {
	content := `
invalid yaml content
  - broken: [
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	_, err := LoadConfig(configPath)
	if err == nil {
		t.Error("LoadConfig() should return error for invalid YAML")
	}
}

func TestLoadConfig_EnvVarOverride_VideoModelPath(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	// Set environment variable
	expectedPath := "/env/video/model/path"
	os.Setenv("WAN_MODEL_PATH", expectedPath)
	defer os.Unsetenv("WAN_MODEL_PATH")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Video.ModelPath != expectedPath {
		t.Errorf("Video.ModelPath = %q, want %q (from env)", cfg.Video.ModelPath, expectedPath)
	}
}

func TestLoadConfig_EnvVarOverride_FaceSwapModelPath(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	// Set environment variable
	expectedPath := "/env/faceswap/model/path"
	os.Setenv("FACESWAP_MODEL_PATH", expectedPath)
	defer os.Unsetenv("FACESWAP_MODEL_PATH")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.FaceSwap.ModelPath != expectedPath {
		t.Errorf("FaceSwap.ModelPath = %q, want %q (from env)", cfg.FaceSwap.ModelPath, expectedPath)
	}
}

func TestLoadConfig_ConfigOverridesEnv(t *testing.T) {
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
video:
  model_path: "/config/video/path"
faceswap:
  model_path: "/config/faceswap/path"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	// Set environment variables (should be ignored if config has value)
	os.Setenv("WAN_MODEL_PATH", "/env/video/path")
	os.Setenv("FACESWAP_MODEL_PATH", "/env/faceswap/path")
	defer os.Unsetenv("WAN_MODEL_PATH")
	defer os.Unsetenv("FACESWAP_MODEL_PATH")

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	// Config values should take precedence over env vars
	if cfg.Video.ModelPath != "/config/video/path" {
		t.Errorf("Video.ModelPath = %q, want %q (config should override env)", cfg.Video.ModelPath, "/config/video/path")
	}
	if cfg.FaceSwap.ModelPath != "/config/faceswap/path" {
		t.Errorf("FaceSwap.ModelPath = %q, want %q (config should override env)", cfg.FaceSwap.ModelPath, "/config/faceswap/path")
	}
}

func TestConfig_NeedsRegistration(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   string
		workerID string
		expected bool
	}{
		{"both empty", "", "", true},
		{"key empty", "", "worker-id", true},
		{"id empty", "wk_key", "", true},
		{"both set", "wk_key", "worker-id", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				API:    APIConfig{Key: tt.apiKey},
				Worker: WorkerConfig{ID: tt.workerID},
			}
			if got := cfg.NeedsRegistration(); got != tt.expected {
				t.Errorf("NeedsRegistration() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestConfig_Validate_Success(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "https://api.gelotto.io",
			Key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		},
		Worker: WorkerConfig{
			ID: "test-worker-id",
		},
	}

	if err := cfg.Validate(); err != nil {
		t.Errorf("Validate() error = %v, want nil", err)
	}
}

func TestConfig_Validate_MissingAPIKey(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "https://api.gelotto.io",
			Key: "",
		},
		Worker: WorkerConfig{
			ID: "test-worker-id",
		},
	}

	err := cfg.Validate()
	if err == nil {
		t.Error("Validate() should return error for missing API key")
	}
	if err.Error() != "API key not configured" {
		t.Errorf("Validate() error = %q, want %q", err.Error(), "API key not configured")
	}
}

func TestConfig_Validate_InvalidKeyPrefix(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "https://api.gelotto.io",
			Key: "sk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", // Wrong prefix
		},
		Worker: WorkerConfig{
			ID: "test-worker-id",
		},
	}

	err := cfg.Validate()
	if err == nil {
		t.Error("Validate() should return error for invalid key prefix")
	}
	if err.Error() != "invalid API key format (should start with 'wk_')" {
		t.Errorf("Validate() error = %q", err.Error())
	}
}

func TestConfig_Validate_InvalidKeyLength(t *testing.T) {
	tests := []struct {
		name   string
		key    string
		wantOk bool
	}{
		{"too short", "wk_short", false},
		{"exact length (67)", "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", true},
		{"too long", "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdefextra", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				API: APIConfig{
					URL: "https://api.gelotto.io",
					Key: tt.key,
				},
				Worker: WorkerConfig{
					ID: "test-worker-id",
				},
			}

			err := cfg.Validate()
			if tt.wantOk && err != nil {
				t.Errorf("Validate() error = %v, want nil", err)
			}
			if !tt.wantOk && err == nil {
				t.Error("Validate() should return error")
			}
		})
	}
}

func TestConfig_Validate_MissingWorkerID(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "https://api.gelotto.io",
			Key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		},
		Worker: WorkerConfig{
			ID: "",
		},
	}

	err := cfg.Validate()
	if err == nil {
		t.Error("Validate() should return error for missing worker ID")
	}
	if err.Error() != "worker ID not configured" {
		t.Errorf("Validate() error = %q, want %q", err.Error(), "worker ID not configured")
	}
}

func TestConfig_Validate_MissingAPIURL(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "",
			Key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		},
		Worker: WorkerConfig{
			ID: "test-worker-id",
		},
	}

	err := cfg.Validate()
	if err == nil {
		t.Error("Validate() should return error for missing API URL")
	}
	if err.Error() != "API URL not configured" {
		t.Errorf("Validate() error = %q, want %q", err.Error(), "API URL not configured")
	}
}

func TestSaveConfig(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			URL: "https://api.gelotto.io",
			Key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		},
		Model: ModelConfig{
			Path:        "/path/to/model",
			Name:        "z-image-turbo",
			ServiceMode: "gguf",
			VRAMGB:      16,
		},
		Worker: WorkerConfig{
			ID:                "saved-worker-id",
			Hostname:          "saved-host",
			PollInterval:      10 * time.Second,
			HeartbeatInterval: 45 * time.Second,
		},
	}

	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "saved-config.yaml")

	// Save config
	if err := SaveConfig(configPath, cfg); err != nil {
		t.Fatalf("SaveConfig() error = %v", err)
	}

	// Verify file exists with correct permissions
	info, err := os.Stat(configPath)
	if err != nil {
		t.Fatalf("Config file not created: %v", err)
	}
	if info.Mode().Perm() != 0600 {
		t.Errorf("Config file permissions = %o, want %o", info.Mode().Perm(), 0600)
	}

	// Load it back and verify
	loaded, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if loaded.API.Key != cfg.API.Key {
		t.Errorf("Loaded API.Key = %q, want %q", loaded.API.Key, cfg.API.Key)
	}
	if loaded.Worker.ID != cfg.Worker.ID {
		t.Errorf("Loaded Worker.ID = %q, want %q", loaded.Worker.ID, cfg.Worker.ID)
	}
	if loaded.Model.VRAMGB != cfg.Model.VRAMGB {
		t.Errorf("Loaded Model.VRAMGB = %d, want %d", loaded.Model.VRAMGB, cfg.Model.VRAMGB)
	}
}

func TestSaveConfig_InvalidPath(t *testing.T) {
	cfg := &Config{}
	err := SaveConfig("/nonexistent/directory/config.yaml", cfg)
	if err == nil {
		t.Error("SaveConfig() should return error for invalid path")
	}
}

func TestVideoConfig_EnabledPointer(t *testing.T) {
	// Test nil (auto-detect)
	content := `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
`
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Video.Enabled != nil {
		t.Error("Video.Enabled should be nil for auto-detect")
	}

	// Test explicit false
	content = `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
video:
  enabled: false
`
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err = LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Video.Enabled == nil {
		t.Fatal("Video.Enabled should not be nil when explicitly set")
	}
	if *cfg.Video.Enabled != false {
		t.Error("Video.Enabled should be false")
	}

	// Test explicit true
	content = `
api:
  key: "wk_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
worker:
  id: "test-worker"
video:
  enabled: true
`
	if err := os.WriteFile(configPath, []byte(content), 0600); err != nil {
		t.Fatalf("Failed to write test config: %v", err)
	}

	cfg, err = LoadConfig(configPath)
	if err != nil {
		t.Fatalf("LoadConfig() error = %v", err)
	}

	if cfg.Video.Enabled == nil {
		t.Fatal("Video.Enabled should not be nil when explicitly set")
	}
	if *cfg.Video.Enabled != true {
		t.Error("Video.Enabled should be true")
	}
}
