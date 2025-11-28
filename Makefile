.PHONY: build run clean help compose-up compose-down compose-rebuild

IMAGE_NAME := slangen-runner
COMPOSE_FILE := docker-compose.yml
SERVICE_NAME := slanggen

help:
	@echo "Available targets:"
	@echo "  make build           - Build the Docker image"
	@echo "  make run             - Run the Docker container"
	@echo "  make clean           - Remove the Docker image and stop running containers"
	@echo "  make compose-up      - Start the app via docker-compose in detached mode"
	@echo "  make compose-down    - Stop the docker-compose services"
	@echo "  make compose-rebuild - Rebuild and start the docker-compose services"

# Build Docker image
build:
	docker build --pull -t $(IMAGE_NAME) .

# Run Docker image
run: build
	docker run --rm -p 8000:8000 $(IMAGE_NAME)

# Clean up Docker images and containers
clean:
	-docker rm -f $(IMAGE_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	-docker builder prune -f 2>/dev/null || true
	-docker image prune -f 2>/dev/null || true

# Test the Docker image
test: build
	docker run --rm $(IMAGE_NAME) pytest --cov -v

# Docker Compose targets
compose-up:
	docker compose -f $(COMPOSE_FILE) up -d --build

compose-down:
	docker compose -f $(COMPOSE_FILE) down

compose-rebuild:
	docker compose -f $(COMPOSE_FILE) down
	docker compose -f $(COMPOSE_FILE) up -d --build
