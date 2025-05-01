.PHONY: all build run test clean ui-setup ui-dev ui-build

all: build

build:
	@echo "Building OpenLLM..."
	go build -o bin/openllm ./cmd/openllm

run: build
	@echo "Running OpenLLM..."
	./bin/openllm

test:
	@echo "Running tests..."
	go test ./...

clean:
	@echo "Cleaning up..."
	rm -rf bin/
	go clean

ui-setup:
	@echo "Setting up UI dependencies..."
	cd ui && npm install

ui-dev:
	@echo "Starting UI development server..."
	cd ui && npm run dev

ui-build:
	@echo "Building UI..."
	cd ui && npm run build 