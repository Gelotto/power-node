package worker

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config holds worker configuration
type Config struct {
	API    APIConfig    `yaml:"api"`
	Model  ModelConfig  `yaml:"model"`
	Video  VideoConfig  `yaml:"video"`
	Worker WorkerConfig `yaml:"worker"`
	Python PythonConfig `yaml:"python"`
}

// APIConfig holds API connection settings
type APIConfig struct {
	URL string `yaml:"url"`
	Key string `yaml:"key"`
}

// ModelConfig holds model settings
type ModelConfig struct {
	Path        string `yaml:"path"`
	Name        string `yaml:"name"`
	ServiceMode string `yaml:"service_mode"`
	VRAMGB      int    `yaml:"vram_gb"`
}

// VideoConfig holds video generation settings
type VideoConfig struct {
	Enabled     *bool  `yaml:"enabled"`      // nil = auto-detect, true/false = explicit
	ModelPath   string `yaml:"model_path"`   // Path to Wan2.1 model (if empty, check WAN_MODEL_PATH env)
	MaxDuration int    `yaml:"max_duration"` // Maximum video duration in seconds (default: 5)
	MaxFPS      int    `yaml:"max_fps"`      // Maximum video FPS (default: 24)
	MaxWidth    int    `yaml:"max_width"`    // Maximum video width (default: 480)
	MaxHeight   int    `yaml:"max_height"`   // Maximum video height (default: 480)
}

// WorkerConfig holds worker-specific settings
type WorkerConfig struct {
	ID                string        `yaml:"id"`
	Hostname          string        `yaml:"hostname"`
	GPUInfo           string        `yaml:"gpu_info"`
	PollInterval      time.Duration `yaml:"poll_interval"`
	HeartbeatInterval time.Duration `yaml:"heartbeat_interval"`
	IdleDetection     IdleConfig    `yaml:"idle_detection"` // GPU idle detection (opt-in)
}

// PythonConfig holds Python interpreter settings
type PythonConfig struct {
	Executable string            `yaml:"executable"`
	ScriptPath string            `yaml:"script_path"`
	ScriptArgs []string          `yaml:"script_args"`
	Env        map[string]string `yaml:"env"`
}

// IdleConfig holds GPU idle detection settings (opt-in feature)
type IdleConfig struct {
	Enabled              bool          `yaml:"enabled"`               // Default: false (opt-in)
	UtilizationThreshold int           `yaml:"utilization_threshold"` // Default: 50%
	MemoryThreshold      int           `yaml:"memory_threshold"`      // Default: 80%
	CheckInterval        time.Duration `yaml:"check_interval"`        // Default: 5s
	CooldownPeriod       time.Duration `yaml:"cooldown_period"`       // Default: 30s
}

// LoadConfig loads configuration from a YAML file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Set defaults
	if cfg.API.URL == "" {
		cfg.API.URL = "https://api.gelotto.io"
	}
	if cfg.Model.Name == "" {
		cfg.Model.Name = "z-image-turbo"
	}
	if cfg.Worker.PollInterval == 0 {
		cfg.Worker.PollInterval = 5 * time.Second
	}
	if cfg.Worker.HeartbeatInterval == 0 {
		cfg.Worker.HeartbeatInterval = 30 * time.Second
	}
	if cfg.Python.Executable == "" {
		cfg.Python.Executable = "python3"
	}
	if cfg.Python.ScriptPath == "" {
		cfg.Python.ScriptPath = "scripts/inference.py"
	}

	// Video config defaults
	if cfg.Video.MaxDuration == 0 {
		cfg.Video.MaxDuration = 5 // 5 seconds max
	}
	if cfg.Video.MaxFPS == 0 {
		cfg.Video.MaxFPS = 24
	}
	if cfg.Video.MaxWidth == 0 {
		cfg.Video.MaxWidth = 480
	}
	if cfg.Video.MaxHeight == 0 {
		cfg.Video.MaxHeight = 480
	}
	// Check WAN_MODEL_PATH env if not set in config
	if cfg.Video.ModelPath == "" {
		cfg.Video.ModelPath = os.Getenv("WAN_MODEL_PATH")
	}

	// Idle detection defaults (feature is opt-in, so enabled defaults to false)
	if cfg.Worker.IdleDetection.UtilizationThreshold == 0 {
		cfg.Worker.IdleDetection.UtilizationThreshold = 50
	}
	if cfg.Worker.IdleDetection.MemoryThreshold == 0 {
		cfg.Worker.IdleDetection.MemoryThreshold = 80
	}
	if cfg.Worker.IdleDetection.CheckInterval == 0 {
		cfg.Worker.IdleDetection.CheckInterval = 5 * time.Second
	}
	if cfg.Worker.IdleDetection.CooldownPeriod == 0 {
		cfg.Worker.IdleDetection.CooldownPeriod = 30 * time.Second
	}

	return &cfg, nil
}

// NeedsRegistration returns true if the worker needs to register
func (c *Config) NeedsRegistration() bool {
	return c.API.Key == "" || c.Worker.ID == ""
}

// Validate checks if the configuration is valid and returns detailed errors
func (c *Config) Validate() error {
	if c.API.Key == "" {
		return fmt.Errorf("API key not configured")
	}
	if !strings.HasPrefix(c.API.Key, "wk_") {
		return fmt.Errorf("invalid API key format (should start with 'wk_')")
	}
	if len(c.API.Key) != 67 {
		return fmt.Errorf("invalid API key length (expected 67 characters, got %d)", len(c.API.Key))
	}
	if c.Worker.ID == "" {
		return fmt.Errorf("worker ID not configured")
	}
	if c.API.URL == "" {
		return fmt.Errorf("API URL not configured")
	}
	return nil
}

// SaveConfig saves the configuration back to a YAML file
func SaveConfig(path string, cfg *Config) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}
