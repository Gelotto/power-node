package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/Gelotto/power-node/internal/client"
	"github.com/Gelotto/power-node/internal/worker"
)

// Version is set at build time via ldflags
var Version = "dev"

func main() {
	configPath := flag.String("config", "", "Path to configuration file")
	statusFlag := flag.Bool("status", false, "Show worker status and exit")
	checkFlag := flag.Bool("check", false, "Validate configuration and exit")
	versionFlag := flag.Bool("version", false, "Show version and exit")
	updateFlag := flag.Bool("update", false, "Update to latest version and exit")
	flag.Parse()

	// Handle --version
	if *versionFlag {
		fmt.Printf("power-node version %s\n", Version)
		os.Exit(0)
	}

	// Handle --update
	if *updateFlag {
		if err := selfUpdate(); err != nil {
			fmt.Fprintf(os.Stderr, "Update failed: %v\n", err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	// Determine config path
	if *configPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			log.Fatalf("Failed to get home directory: %v", err)
		}
		*configPath = filepath.Join(homeDir, ".power-node", "config", "config.yaml")
	}

	// Handle --status
	if *statusFlag {
		showStatus(*configPath)
		os.Exit(0)
	}

	// Handle --check
	if *checkFlag {
		if runChecks(*configPath) {
			os.Exit(0)
		}
		os.Exit(1)
	}

	// Normal startup
	log.Printf("Loading configuration from %s...", *configPath)
	config, err := worker.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	w, err := worker.NewWorker(config, *configPath)
	if err != nil {
		log.Fatalf("Failed to create worker: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	errChan := make(chan error, 1)
	go func() {
		if err := w.Start(ctx); err != nil {
			errChan <- err
		}
	}()

	select {
	case sig := <-sigChan:
		log.Printf("Received signal: %v, shutting down...", sig)
		cancel()
		if err := w.Stop(); err != nil {
			log.Printf("Error during shutdown: %v", err)
		}

	case err := <-errChan:
		log.Printf("Worker error: %v", err)
		cancel()
		_ = w.Stop()
		os.Exit(1)
	}

	log.Println("Worker shutdown complete")
}

func showStatus(configPath string) {
	fmt.Println("Power Node Status")
	fmt.Println("─────────────────")
	fmt.Printf("Version:    %s\n", Version)
	fmt.Printf("Config:     %s\n", configPath)

	// Check if config exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		fmt.Println("\n✗ Config file not found")
		fmt.Printf("\n  Run the installer first:\n")
		fmt.Printf("  curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash\n")
		return
	}

	config, err := worker.LoadConfig(configPath)
	if err != nil {
		fmt.Printf("\n✗ Failed to load config: %v\n", err)
		return
	}

	// Worker ID
	if config.Worker.ID != "" {
		fmt.Printf("Worker ID:  %s\n", config.Worker.ID)
	} else {
		fmt.Printf("Worker ID:  (not configured)\n")
	}

	// API Key
	if config.API.Key != "" {
		keyPreview := config.API.Key
		if len(keyPreview) > 15 {
			keyPreview = keyPreview[:15] + "..."
		}
		fmt.Printf("API Key:    %s (configured)\n", keyPreview)
	} else {
		fmt.Printf("API Key:    (not configured)\n")
	}

	fmt.Printf("API URL:    %s\n", config.API.URL)

	// GPU Info
	gpuInfo := detectGPU()
	fmt.Printf("GPU:        %s\n", gpuInfo)

	if config.Model.VRAMGB > 0 {
		fmt.Printf("VRAM:       %d GB\n", config.Model.VRAMGB)
	}

	if config.Model.ServiceMode != "" {
		fmt.Printf("Mode:       %s\n", strings.ToUpper(config.Model.ServiceMode))
	}

	// API connectivity
	fmt.Println("\nConnectivity:")
	apiClient := client.NewAPIClient(config.API.URL, config.API.Key)
	ctx := context.Background()
	if err := apiClient.HealthCheck(ctx); err != nil {
		fmt.Printf("  API Health: ✗ %v\n", err)
	} else {
		fmt.Printf("  API Health: ✓ OK\n")
	}
}

func runChecks(configPath string) bool {
	allPassed := true

	// Check config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		fmt.Printf("✗ Config file not found: %s\n", configPath)
		fmt.Printf("\n  Run the installer first:\n")
		fmt.Printf("  curl -sSL https://raw.githubusercontent.com/Gelotto/power-node/main/install.sh | bash\n")
		return false
	}
	fmt.Println("✓ Config file found")

	// Load config
	config, err := worker.LoadConfig(configPath)
	if err != nil {
		fmt.Printf("✗ Failed to load config: %v\n", err)
		return false
	}
	fmt.Println("✓ Config file valid")

	// Validate config
	if err := config.Validate(); err != nil {
		fmt.Printf("✗ %v\n", err)
		fmt.Printf("\n  Please add your credentials to:\n")
		fmt.Printf("  %s\n\n", configPath)
		fmt.Printf("  api:\n")
		fmt.Printf("    key: \"YOUR_API_KEY_HERE\"\n\n")
		fmt.Printf("  worker:\n")
		fmt.Printf("    id: \"YOUR_WORKER_ID_HERE\"\n\n")
		fmt.Printf("  Get your credentials at: https://gelotto.io/workers\n")
		return false
	}
	fmt.Println("✓ API key configured")
	fmt.Println("✓ Worker ID configured")

	// Check API connectivity
	apiClient := client.NewAPIClient(config.API.URL, config.API.Key)
	ctx := context.Background()
	if err := apiClient.HealthCheck(ctx); err != nil {
		fmt.Printf("✗ API connectivity failed: %v\n", err)
		allPassed = false
	} else {
		fmt.Println("✓ API connectivity OK")
	}

	// Check GPU
	gpuInfo := detectGPU()
	if gpuInfo == "Unknown GPU" || gpuInfo == "" {
		fmt.Println("✗ GPU not detected (nvidia-smi not found or no GPU)")
		allPassed = false
	} else {
		fmt.Printf("✓ GPU detected: %s\n", gpuInfo)
	}

	if allPassed {
		fmt.Println("\nAll checks passed!")
	} else {
		fmt.Println("\nSome checks failed. Please resolve the issues above.")
	}

	return allPassed
}

func detectGPU() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return "Unknown GPU"
	}
	gpu := strings.TrimSpace(string(out))
	if gpu == "" {
		return "Unknown GPU"
	}
	// Return first GPU if multiple
	if idx := strings.Index(gpu, "\n"); idx > 0 {
		gpu = gpu[:idx]
	}
	return gpu
}

func selfUpdate() error {
	fmt.Printf("Current version: %s\n", Version)
	fmt.Println("Checking for updates...")

	// Determine binary URL
	osName := runtime.GOOS
	arch := runtime.GOARCH
	if arch == "amd64" {
		arch = "x86_64"
	}

	binaryURL := fmt.Sprintf("https://github.com/Gelotto/power-node/releases/latest/download/power-node-%s-%s", osName, arch)

	// Get current executable path
	execPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("failed to get executable path: %w", err)
	}
	execPath, err = filepath.EvalSymlinks(execPath)
	if err != nil {
		return fmt.Errorf("failed to resolve executable path: %w", err)
	}

	// Download new binary
	fmt.Printf("Downloading from %s...\n", binaryURL)
	resp, err := http.Get(binaryURL)
	if err != nil {
		return fmt.Errorf("failed to download update: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("no release found for %s-%s", osName, arch)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Write to temp file
	tmpPath := execPath + ".new"
	tmpFile, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}

	_, err = io.Copy(tmpFile, resp.Body)
	_ = tmpFile.Close()
	if err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to write update: %w", err)
	}

	// Backup old binary
	backupPath := execPath + ".old"
	_ = os.Remove(backupPath) // Remove old backup if exists
	if err := os.Rename(execPath, backupPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to backup current binary: %w", err)
	}

	// Move new binary into place
	if err := os.Rename(tmpPath, execPath); err != nil {
		// Try to restore backup
		_ = os.Rename(backupPath, execPath)
		return fmt.Errorf("failed to install update: %w", err)
	}

	// Get new version
	out, err := exec.Command(execPath, "--version").Output()
	if err != nil {
		fmt.Println("Update installed! Restart power-node to use new version.")
	} else {
		newVersion := strings.TrimSpace(string(out))
		fmt.Printf("Updated to %s\n", newVersion)
	}

	fmt.Println("\nIf running as a service, restart with:")
	fmt.Println("  sudo systemctl restart power-node")

	return nil
}
