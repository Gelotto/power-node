package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/Gelotto/power-node/internal/worker"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()

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
		w.Stop()
		os.Exit(1)
	}

	log.Println("Worker shutdown complete")
}
