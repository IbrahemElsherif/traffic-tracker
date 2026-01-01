# Contributing to Traffic Tracker

Thank you for your interest in contributing to Traffic Tracker! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-tracker.git
   cd traffic-tracker
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

This project follows:
- **Black** for code formatting (line length: 88)
- **isort** for import sorting (black profile)
- **flake8** for linting

Run formatters before committing:
```bash
black src tests app.py
isort src tests app.py
flake8 src tests app.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py -v

# Skip integration tests
pytest -m "not integration"
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Commit Message Format

Use clear, descriptive commit messages:
```
feat: add support for custom vehicle classes
fix: resolve memory leak in video processing
docs: update installation instructions
test: add unit tests for config module
refactor: simplify frame processing logic
```

## Questions?

Feel free to open an issue for questions or discussions.

