#!/bin/bash

# Script to build Docker images for all task environments in datasets/terminal-bench-2
# and push them to GitHub Container Registry (GHCR)
# Requires Docker to be installed and authenticated with GHCR (ghcr.io)
# x86-64 native architecture

set -e  # Exit on error

DATASETS_DIR="datasets/terminal-bench-2"
REGISTRY="ghcr.io"
OWNER="vals-ai"
TARGET_TASK=""
PARALLEL_JOBS=5

# Get version from git commit hash, or use date if not in a git repo
if git rev-parse --short HEAD &> /dev/null; then
    VERSION=$(git rev-parse --short HEAD)
else
    VERSION=$(date +%Y%m%d)
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [TASK_ID] [--parallel JOBS]"
            echo ""
            echo "Build and push Docker images to GitHub Container Registry"
            echo ""
            echo "Arguments:"
            echo "  (none)            Build and push all tasks"
            echo "  TASK_ID           Build and push only the specified task (e.g., 'build-pov-ray')"
            echo "  --parallel JOBS   Number of concurrent builds (default: 5)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Build all environments (5 concurrent)"
            echo "  $0 build-pov-ray            # Test with just build-pov-ray"
            echo "  $0 --parallel 3             # Build all with 3 concurrent jobs"
            exit 0
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            TARGET_TASK="$1"
            shift
            ;;
    esac
done

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
echo "Registry: $REGISTRY/$OWNER"
echo "Version: $VERSION"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Function to build and push a single task
build_task() {
    local task_dir=$1
    local task_id=$2
    local current=$3
    local total=$4

    # Skip if no Dockerfile exists
    if [ ! -f "$task_dir/environment/Dockerfile" ]; then
        echo "[$current/$total] SKIP: $task_id (no Dockerfile found)"
        return 0
    fi

    echo "[$current/$total] Building and pushing: $task_id"

    # Build and push the Docker image
    # Using GHCR image naming convention: ghcr.io/owner/task-id:tag
    # --platform linux/amd64 specifies x86-64 architecture
    base_image="$REGISTRY/$OWNER/$task_id"
    image_latest="$base_image:latest"
    image_versioned="$base_image:$VERSION"

    if docker build \
        --platform linux/amd64 \
        --label "org.opencontainers.image.source=https://github.com/$OWNER/terminal-bench-benchmark-service" \
        --label "org.opencontainers.image.url=https://github.com/$OWNER/terminal-bench-benchmark-service" \
        --label "org.opencontainers.image.version=$VERSION" \
        -f "$task_dir/environment/Dockerfile" \
        -t "$image_latest" \
        -t "$image_versioned" \
        "$task_dir/environment" > /tmp/build_$task_id.log 2>&1; then
        echo "  ✓ Successfully built $task_id"

        # Push both tags to GHCR
        local push_failed=0
        if docker push "$image_latest" > /tmp/push_$task_id.log 2>&1; then
            echo "  ✓ Pushed $image_latest"
        else
            echo "  ✗ Failed to push $image_latest"
            cat /tmp/push_$task_id.log
            push_failed=1
        fi

        if docker push "$image_versioned" >> /tmp/push_$task_id.log 2>&1; then
            echo "  ✓ Pushed $image_versioned"
        else
            echo "  ✗ Failed to push $image_versioned"
            cat /tmp/push_$task_id.log
            push_failed=1
        fi

        if [ $push_failed -eq 1 ]; then
            return 1
        fi
    else
        echo "  ✗ Failed to build $task_id"
        cat /tmp/build_$task_id.log
        return 1
    fi
}

export -f build_task
export REGISTRY OWNER REPO

# Run builds in parallel
active_jobs=0
failed_tasks=()

for task_dir in "${task_dirs[@]}"; do
    task_id=$(basename "$task_dir")
    current=$((current + 1))

    # Start build in background
    build_task "$task_dir" "$task_id" "$current" "$total" &
    active_jobs=$((active_jobs + 1))

    # Wait for a job to finish if we've reached max parallel jobs
    if [ $active_jobs -ge $PARALLEL_JOBS ]; then
        wait -n
        active_jobs=$((active_jobs - 1))
    fi
done

# Wait for remaining jobs to finish
wait

echo ""
echo "Build and push process complete!"
echo "Images are available at: $REGISTRY/$OWNER/<task-id>"
echo ""
echo "Each image is tagged with:"
echo "  - latest: $REGISTRY/$OWNER/<task-id>:latest"
echo "  - versioned: $REGISTRY/$OWNER/<task-id>:$VERSION"
echo ""
echo "Example: $REGISTRY/$OWNER/build-pov-ray:latest"
