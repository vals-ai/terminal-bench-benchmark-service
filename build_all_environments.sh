#!/bin/bash

# Script to build Docker images for all task environments in datasets/terminal-bench-2
# and push them to GitHub Container Registry (GHCR)
# Requires Docker to be installed and authenticated with GHCR (ghcr.io)
# x86-64 native architecture

set -e  # Exit on error

DATASETS_DIR="datasets/terminal-bench-2"
REGISTRY="ghcr.io"
OWNER="vals-ai"
REPO="terminal-bench-benchmark-service"
TARGET_TASK="${1:-}"  # Optional: specific task to build, or empty for all

# Print usage if help is requested
if [ "$TARGET_TASK" = "-h" ] || [ "$TARGET_TASK" = "--help" ]; then
    echo "Usage: $0 [TASK_ID]"
    echo ""
    echo "Build and push Docker images to GitHub Container Registry"
    echo ""
    echo "Arguments:"
    echo "  (none)     Build and push all tasks"
    echo "  TASK_ID    Build and push only the specified task (e.g., 'build-pov-ray')"
    echo "  -h, --help Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build all environments"
    echo "  $0 build-pov-ray      # Test with just build-pov-ray"
    exit 0
fi

if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: $DATASETS_DIR directory not found"
    exit 1
fi

# Check if user is authenticated with GHCR
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "Warning: You may not be authenticated with GHCR"
    echo "Login with: docker login ghcr.io"
    echo ""
fi

# Determine which tasks to build
if [ -n "$TARGET_TASK" ]; then
    # Build only a specific task
    if [ ! -d "$DATASETS_DIR/$TARGET_TASK" ]; then
        echo "Error: Task '$TARGET_TASK' not found in $DATASETS_DIR"
        exit 1
    fi
    task_dirs=("$DATASETS_DIR/$TARGET_TASK/")
    echo "Testing with single task: $TARGET_TASK"
else
    # Build all tasks
    task_dirs=("$DATASETS_DIR"/*/)
fi

total=${#task_dirs[@]}
current=0

echo "Found $total task environment(s) to build and push"
echo "Registry: $REGISTRY/$OWNER/$REPO"
echo ""

# Iterate through each task directory
for task_dir in "${task_dirs[@]}"; do
    task_id=$(basename "$task_dir")
    current=$((current + 1))

    # Skip if no Dockerfile exists
    if [ ! -f "$task_dir/environment/Dockerfile" ]; then
        echo "[$current/$total] SKIP: $task_id (no Dockerfile found)"
        continue
    fi

    echo "[$current/$total] Building and pushing: $task_id"

    # Build and push the Docker image
    # Using GHCR image naming convention
    # --platform linux/amd64 specifies x86-64 architecture
    image_name="$REGISTRY/$OWNER/$REPO:$task_id"

    if (cd "$task_dir" && docker build \
        --platform linux/amd64 \
        -f environment/Dockerfile \
        -t "$image_name" \
        .); then
        echo "  ✓ Successfully built $task_id"

        # Push to GHCR
        if docker push "$image_name"; then
            echo "  ✓ Successfully pushed $image_name"
        else
            echo "  ✗ Failed to push $image_name"
        fi
    else
        echo "  ✗ Failed to build $task_id"
    fi
    echo ""
done

echo "Build and push process complete!"
echo "Images are available at: $REGISTRY/$OWNER/$REPO"
