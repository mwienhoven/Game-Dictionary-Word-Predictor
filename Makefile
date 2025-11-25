.PHONY: build run clean help

IMAGE_NAME := slangen-runner

help:
	@echo "Available targets:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Run the Docker container"
	@echo "  make clean  - Remove the Docker image and stop running containers"

build:
	docker build --pull -t $(IMAGE_NAME) .

run: build
	docker run --rm -p 8000:8000 $(IMAGE_NAME)

clean:
	-docker rm -f $(IMAGE_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	-docker builder prune -f 2>/dev/null || true
	-docker image prune -f 2>/dev/null || true
