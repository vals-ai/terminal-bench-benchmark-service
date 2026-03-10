#!/bin/bash

# Script to build Docker images for all task environments in datasets/terminal-bench-2
# Requires Docker to be installed and x86-64 native architecture

set -e  # Exit on error

DATASETS_DIR="datasets/terminal-bench-2"

if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: $DATASETS_DIR directory not found"
    exit 1
fi

# Count total tasks
total=$(find "$DATASETS_DIR" -maxdepth 1 -type d -not -name "terminal-bench-2" | wc -l)
current=0

echo "Found $total task environments to build"
echo ""

# Iterate through each task directory
for task_dir in "$DATASETS_DIR"/*/; do
    task_id=$(basename "$task_dir")
    current=$((current + 1))

    # Skip if no Dockerfile exists
    if [ ! -f "$task_dir/environment/Dockerfile" ]; then
        echo "[$current/$total] SKIP: $task_id (no Dockerfile found)"
        continue
    fi

    echo "[$current/$total] Building: $task_id"

    # Build the Docker image
    # Using the task_id as the image tag
    # --platform linux/amd64 specifies x86-64 architecture
    if docker build \
        --platform linux/amd64 \
        -f environment/Dockerfile \
        -t "terminal-bench-2:$task_id" \
        "$task_dir"; then
        echo "  ✓ Successfully built $task_id"
    else
        echo "  ✗ Failed to build $task_id"
        # Comment out the next line if you want to continue despite build failures
        # exit 1
    fi
    echo ""
done

echo "Build process complete!"
