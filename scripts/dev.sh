#!/bin/bash
# Development workflow for AegisAV using uv

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Setup development environment
setup_env() {
    print_status "Setting up development environment..."

    # Check if .venv exists
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        uv venv
    fi

    # Activate environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate

    # Install dependencies
    print_status "Installing dependencies..."
    uv pip install -e ".[dev]"

    print_success "Development environment is ready!"
}

# Install production dependencies
install_prod() {
    print_status "Installing production dependencies..."
    uv pip install -e .
    print_success "Production dependencies installed!"
}

# Run tests
run_tests() {
    print_status "Running tests..."

    # Check if we're in the right environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Please activate the virtual environment first: source .venv/bin/activate"
        exit 1
    fi

    # Run tests with coverage
    python -m pytest tests/ -v --cov=agent --cov=autonomy --cov=metrics --cov-report=html --cov-report=xml

    print_success "Tests completed!"
}

# Run linting
run_lint() {
    print_status "Running linter..."

    # Run ruff
    ruff check .
    ruff format . --diff

    print_success "Linting completed!"
}

# Run type checking
run_typecheck() {
    print_status "Running type checks..."
    mypy agent autonomy metrics
    print_success "Type checking completed!"
}

# Run all checks
run_all_checks() {
    print_status "Running all quality checks..."
    run_lint
    run_typecheck
    run_tests
    print_success "All checks completed!"
}

# Start development server
run_server() {
    print_status "Starting agent server..."

    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Please activate the virtual environment first: source .venv/bin/activate"
        exit 1
    fi

    # Start the server
    aegis-server
}

# Run demo (without SITL)
run_demo() {
    print_status "Running demo scenario..."

    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Please activate the virtual environment first: source .venv/bin/activate"
        exit 1
    fi

    # Run demo
    python scripts/run_agent.py --scenario basic --demo
}

# Install and setup for SITL simulation
setup_sim() {
    print_status "Setting up simulation environment..."

    # Check if ArduPilot tools are available
    if ! command -v sim_vehicle.py &> /dev/null; then
        print_warning "ArduPilot SITL tools not found. Simulation may not work."
        print_status "Install ArduPilot tools with: sudo apt-get install ardupilot-tools"
        print_status "Or follow: https://ardupilot.org/dev/docs/sitl-software-in-the-simulator"
    fi

    print_success "Simulation setup completed!"
}

# Clean build artifacts
clean() {
    print_status "Cleaning build artifacts..."

    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Remove coverage artifacts
    rm -rf htmlcov/
    rm -f coverage.xml
    rm -f .coverage

    # Remove build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/

    # Remove pytest cache
    rm -rf .pytest_cache/

    print_success "Clean completed!"
}

# Build for distribution
build() {
    print_status "Building distribution..."

    # First clean
    clean

    # Build
    python -m build

    print_success "Build completed!"
}

# Show help
show_help() {
    echo "AegisAV Development Workflow"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Setup development environment"
    echo "  install   - Install production dependencies"
    echo "  test      - Run tests with coverage"
    echo "  lint      - Run linting and formatting checks"
    echo "  typecheck - Run type checking"
    echo "  check     - Run all quality checks (lint + typecheck + test)"
    echo "  server    - Start agent server"
    echo "  demo      - Run demo scenario"
    echo "  sim       - Setup simulation environment"
    echo "  clean     - Clean build artifacts"
    echo "  build     - Build for distribution"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Initial setup"
    echo "  $0 check     # Run all checks before committing"
    echo "  $0 test      # Just run tests"
    echo "  $0 server    # Start development server"
}

# Main script logic
main() {
    # Check if uv is available
    check_uv

    # Handle commands
    case "$1" in
        "setup")
            setup_env
            ;;
        "install")
            install_prod
            ;;
        "test")
            run_tests
            ;;
        "lint")
            run_lint
            ;;
        "typecheck")
            run_typecheck
            ;;
        "check")
            run_all_checks
            ;;
        "server")
            run_server
            ;;
        "demo")
            run_demo
            ;;
        "sim")
            setup_sim
            ;;
        "clean")
            clean
            ;;
        "build")
            build
            ;;
        "help"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
