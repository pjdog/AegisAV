# Contributing to AegisAV

Thank you for your interest in contributing to AegisAV! This document provides guidelines for development and contribution.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Node.js 18+ (for dashboard development)
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/pjdog/AegisAV.git
cd AegisAV

# Install Python dependencies
uv sync

# Install dashboard dependencies
cd frontend && npm install && cd ..

# Run tests to verify setup
uv run pytest
```

## Project Structure

```
AegisAV/
├── agent/                  # Core AI agent
│   ├── server/             # FastAPI decision server
│   │   ├── critics/        # Multi-critic validation
│   │   ├── vision/         # Vision service integration
│   │   └── monitoring/     # Observability
│   └── client/             # Execution layer
├── autonomy/               # Vehicle interface
├── vision/                 # Computer vision pipeline
├── simulation/             # AirSim + ArduPilot integration
├── frontend/               # React dashboard
├── tests/                  # Test suite
└── examples/               # Demo scripts
```

## Code Style

### Python

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Format with `ruff format`
- Lint with `ruff check`

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

### TypeScript (Dashboard)

- Use ESLint + Prettier
- Follow React best practices

```bash
cd frontend
npm run lint
npm run format
```

## Testing

### Run All Tests

```bash
uv run pytest
```

### Run Specific Tests

```bash
# Run a specific test file
uv run pytest tests/test_advanced_decision.py -v

# Run tests matching a pattern
uv run pytest -k "test_goal" -v

# Run with coverage
uv run pytest --cov=agent --cov=autonomy --cov=vision
```

### Test Categories

| Directory | Purpose |
|-----------|---------|
| `tests/` | Unit and integration tests |
| `tests/integration/` | Full system integration tests |
| `tests/vision/` | Vision pipeline tests |

## Branching Strategy

- `main` - Stable, production-ready code
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Creating a Feature Branch

```bash
git checkout main
git pull
git checkout -b feature/my-feature
```

## Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `test` - Tests
- `refactor` - Code refactoring
- `chore` - Maintenance

Examples:
```
feat(vision): add thermal camera support
fix(critics): handle edge case in safety validation
docs(readme): update simulation instructions
```

## Pull Requests

1. Create a feature branch
2. Make your changes
3. Run tests locally
4. Push and create a PR
5. Wait for review

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

## Architecture Decisions

### Adding New Critics

Critics validate goals before execution. To add a new critic:

1. Create a new file in `agent/server/critics/`
2. Inherit from `BaseCritic`
3. Implement `evaluate()` method
4. Register in `CriticOrchestrator`

```python
from agent.server.critics.base import BaseCritic, CriticResult

class MyCritic(BaseCritic):
    async def evaluate(self, goal, context) -> CriticResult:
        # Validation logic
        return CriticResult(
            approved=True,
            confidence=0.9,
            reasoning="Explanation here"
        )
```

### Adding Vision Detectors

To add a new detection model:

1. Create in `vision/models/`
2. Implement the `BaseDetector` interface
3. Register in `VisionService`

### Adding Simulation Environments

See `simulation/UNREAL_SETUP_GUIDE.md` for detailed instructions on creating new Unreal environments.

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide reproduction steps for bugs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
