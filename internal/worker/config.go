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
