#!/bin/bash
# Batch script wrapper for publishing all nanopublications

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Nanopublication Batch Publisher"
echo "=========================================="
echo ""

# Default values
PUBLISH_MODE=""
DRY_RUN=false
INCLUDE_DIRS=""
LIMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --publish)
            PUBLISH_MODE="$2"
            shift 2
            ;;
        --test)
            PUBLISH_MODE="test"
            shift
            ;;
        --prod)
            PUBLISH_MODE="prod"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --include)
            INCLUDE_DIRS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Publish to test server"
            echo "  --prod              Publish to production server"
            echo "  --publish MODE      Publish mode: test or prod"
            echo "  --dry-run           Show what would be published (no actual publishing)"
            echo "  --include DIRS      Only include files in these directories (space-separated)"
            echo "  --limit N           Limit to first N files"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # Preview what would be published"
            echo "  $0 --test                       # Publish to test server"
            echo "  $0 --prod                       # Publish to production"
            echo "  $0 --test --include npi-waves/  # Only publish npi-waves files"
            echo "  $0 --test --limit 5             # Test with first 5 files"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Try to activate virtual environment if it exists
PYTHON_CMD="python3"
# Check from nanopubs/ directory first (one level up)
if [ -f "../venv/bin/activate" ]; then
    echo "Activating virtual environment: ../venv"
    source ../venv/bin/activate
    PYTHON_CMD="../venv/bin/python3"
# Check from project root (two levels up)
elif [ -f "../../venv/bin/activate" ]; then
    echo "Activating virtual environment: ../../venv"
    source ../../venv/bin/activate
    PYTHON_CMD="../../venv/bin/python3"
# Check if already activated
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using existing virtual environment: $VIRTUAL_ENV"
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
fi

# Check if nanopub is installed using the correct Python
if ! $PYTHON_CMD -c "import nanopub" 2>/dev/null; then
    echo "ERROR: nanopub library not installed"
    echo ""
    echo "Install with:"
    echo "  pip install nanopub rdflib"
    echo ""
    echo "Using Python: $PYTHON_CMD"
    echo "Make sure your virtual environment is activated, or install globally."
    exit 1
fi

# Build command (use the Python from venv if available)
CMD="$PYTHON_CMD publish_all_nanopubs.py"

if [ -n "$PUBLISH_MODE" ]; then
    CMD="$CMD --publish $PUBLISH_MODE"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$INCLUDE_DIRS" ]; then
    CMD="$CMD --include $INCLUDE_DIRS"
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Show what will be done
echo "Configuration:"
if [ -n "$PUBLISH_MODE" ]; then
    echo "  Server: $PUBLISH_MODE"
else
    echo "  Mode: Dry-run (preview only)"
fi

if [ -n "$INCLUDE_DIRS" ]; then
    echo "  Include: $INCLUDE_DIRS"
else
    echo "  Include: waves/ and planform_*/ (complete set)"
    echo "  Note: npi-waves/ excluded by default (subset/alternative versions)"
fi

if [ -n "$LIMIT" ]; then
    echo "  Limit: $LIMIT files"
fi

echo ""
echo "Running command: $CMD"
echo ""
echo "=========================================="
echo ""

# Run the command
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Batch publishing completed!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Batch publishing failed (exit code: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

