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
	API      APIConfig      `yaml:"api"`
	Models   ModelsConfig   `yaml:"models"`   // NEW: Multi-model support
	Model    ModelConfig    `yaml:"model"`    // LEGACY: Kept for backwards compatibility
	Video    VideoConfig    `yaml:"video"`
	FaceSwap FaceSwapConfig `yaml:"faceswap"`
	Worker   WorkerConfig   `yaml:"worker"`
	Python   PythonConfig   `yaml:"python"`
}

// APIConfig holds API connection settings
type APIConfig struct {
	URL string `yaml:"url"`
	Key string `yaml:"key"`
}

// ModelConfig holds model settings (LEGACY - kept for backwards compatibility)
type ModelConfig struct {
	Path        string `yaml:"path"`
	Name        string `yaml:"name"`
	ServiceMode string `yaml:"service_mode"`
	VRAMGB      int    `yaml:"vram_gb"`
}

// ModelDefinition defines a single image generation model for multi-model support
type ModelDefinition struct {
	Name         string `yaml:"name"`          // e.g., "z-image-turbo", "flux-schnell"
	Path         string `yaml:"path"`          // Path to model files
	VRAMRequired int    `yaml:"vram_required"` // VRAM needed when loaded (GB)
	Priority     int    `yaml:"priority"`      // Lower = higher priority for staying loaded (0 = highest)
	Enabled      *bool  `yaml:"enabled"`       // nil = auto-detect based on VRAM, true/false = explicit
}

// ModelsConfig holds multi-model settings
type ModelsConfig struct {
	Models      []ModelDefinition `yaml:"models"`       // List of available models
	IdleTimeout time.Duration     `yaml:"idle_timeout"` // How long a model stays loaded when idle (default: 5min)
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

// FaceSwapConfig holds face-swap settings
type FaceSwapConfig struct {
	Enabled     *bool  `yaml:"enabled"`     // nil = auto-detect, true/false = explicit
	ModelPath   string `yaml:"model_path"`  // Path to face-swap models (if empty, check FACESWAP_MODEL_PATH env)
	MaxFrames   int    `yaml:"max_frames"`  // Maximum GIF frames (default: 100)
	MaxWidth    int    `yaml:"max_width"`   // Maximum image width (default: 2048)
	MaxHeight   int    `yaml:"max_height"`  // Maximum image height (default: 2048)
	Enhancement *bool  `yaml:"enhancement"` // Enable GFPGAN enhancement (default: true)
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
	Executable       string            `yaml:"executable"`
	ScriptPath       string            `yaml:"script_path"`
	ScriptArgs       []string          `yaml:"script_args"`
	Env              map[string]string `yaml:"env"`
	ModelLoadTimeout time.Duration     `yaml:"model_load_timeout"` // Timeout for model loading (default: 60s, max: 120s)
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

	// Multi-model migration: if new format is empty but old format exists, migrate
	if len(cfg.Models.Models) == 0 && cfg.Model.Name != "" {
		cfg.Models.Models = []ModelDefinition{{
			Name:         cfg.Model.Name,
			Path:         cfg.Model.Path,
			VRAMRequired: cfg.Model.VRAMGB,
			Priority:     0,
		}}
	}

	// Default idle timeout for multi-model
	if cfg.Models.IdleTimeout == 0 {
		cfg.Models.IdleTimeout = 5 * time.Minute
	}

	// Legacy single-model default (for backwards compatibility)
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
	if cfg.Python.ModelLoadTimeout == 0 {
		cfg.Python.ModelLoadTimeout = 60 * time.Second // Default: 60s for model loading
	} else if cfg.Python.ModelLoadTimeout > 120*time.Second {
		cfg.Python.ModelLoadTimeout = 120 * time.Second // Max: 120s
	}

	// Video config defaults
	if cfg.Video.MaxDuration == 0 {
		cfg.Video.MaxDuration = 5 // 5 seconds max
	}
	if cfg.Video.MaxFPS == 0 {
		cfg.Video.MaxFPS = 24
	}
	if cfg.Video.MaxWidth == 0 {
		cfg.Video.MaxWidth = 832 // 480p widescreen (16:9 aspect ratio)
	}
	if cfg.Video.MaxHeight == 0 {
		cfg.Video.MaxHeight = 480
	}
	// Check WAN_MODEL_PATH env if not set in config
	if cfg.Video.ModelPath == "" {
		cfg.Video.ModelPath = os.Getenv("WAN_MODEL_PATH")
	}

	// Face-swap config defaults
	if cfg.FaceSwap.MaxFrames == 0 {
		cfg.FaceSwap.MaxFrames = 100
	}
	if cfg.FaceSwap.MaxWidth == 0 {
		cfg.FaceSwap.MaxWidth = 2048
	}
	if cfg.FaceSwap.MaxHeight == 0 {
		cfg.FaceSwap.MaxHeight = 2048
	}
	if cfg.FaceSwap.Enhancement == nil {
		defaultTrue := true
		cfg.FaceSwap.Enhancement = &defaultTrue
	}
	// Check FACESWAP_MODEL_PATH env if not set in config
	if cfg.FaceSwap.ModelPath == "" {
		cfg.FaceSwap.ModelPath = os.Getenv("FACESWAP_MODEL_PATH")
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

// GetSupportedModelNames returns a list of all supported model names
func (c *Config) GetSupportedModelNames() []string {
	names := make([]string, 0, len(c.Models.Models))
	for _, m := range c.Models.Models {
		// Skip explicitly disabled models
		if m.Enabled != nil && !*m.Enabled {
			continue
		}
		names = append(names, m.Name)
	}
	// Fallback to legacy single-model if no models configured
	if len(names) == 0 && c.Model.Name != "" {
		return []string{c.Model.Name}
	}
	return names
}

// GetModelDefinition returns the model definition for a given name, or nil if not found
func (c *Config) GetModelDefinition(name string) *ModelDefinition {
	for i := range c.Models.Models {
		if c.Models.Models[i].Name == name {
			return &c.Models.Models[i]
		}
	}
	return nil
}

// HasModel checks if the worker supports a specific model
func (c *Config) HasModel(name string) bool {
	return c.GetModelDefinition(name) != nil
}

// GetDefaultModelName returns the highest-priority model name (lowest priority number)
func (c *Config) GetDefaultModelName() string {
	if len(c.Models.Models) == 0 {
		return c.Model.Name // Legacy fallback
	}

	// Find lowest priority (highest precedence) model
	best := &c.Models.Models[0]
	for i := range c.Models.Models {
		m := &c.Models.Models[i]
		// Skip disabled models
		if m.Enabled != nil && !*m.Enabled {
			continue
		}
		if m.Priority < best.Priority {
			best = m
		}
	}
	return best.Name
}

// IsMultiModel returns true if multiple models are configured
func (c *Config) IsMultiModel() bool {
	enabled := 0
	for _, m := range c.Models.Models {
		if m.Enabled == nil || *m.Enabled {
			enabled++
		}
	}
	return enabled > 1
}
