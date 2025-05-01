# Contributing to OpenLLM

First off, thank you for considering contributing to OpenLLM! It's people like you that make OpenLLM such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please report unacceptable behavior to project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include stack traces and error messages
* Include system information (Go version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful
* List some other LLM projects or applications where this enhancement exists

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Go coding style
* Include appropriate test coverage
* Update documentation as needed
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. Run `go mod tidy` to ensure dependencies are correct
3. If you've added code that should be tested, add tests
4. Ensure the test suite passes
5. Make sure your code follows the Go formatting guidelines
6. Issue that pull request!

### Development Setup

1. Install Go 1.21 or later
2. Clone your fork of the repo
3. Run `go mod download` to install dependencies
4. Create a branch for your changes

### Coding Style

* Follow standard Go formatting (use `go fmt`)
* Write descriptive commit messages
* Include comments for non-obvious code sections
* Add documentation for public APIs
* Keep functions focused and modular

### Testing

* Write unit tests for new code
* Ensure all tests pass locally before submitting
* Include integration tests for new features
* Test edge cases and error conditions

## Project Structure

```
openllm/
├── pkg/
│   ├── model/           # Core transformer implementation
│   ├── tensor/          # Tensor operations
│   ├── training/        # Training pipeline
│   ├── optimization/    # Model optimizations
│   └── metrics/         # Evaluation metrics
├── cmd/                 # Command-line tools
├── docs/                # Documentation
└── tests/              # Integration tests
```

## Documentation

* Update README.md with details of changes to the interface
* Update API documentation for changed endpoints
* Add examples for new features
* Update the changelog

## Review Process

The core team looks at Pull Requests on a regular basis. After feedback has been given we expect responses within two weeks. After two weeks we may close the PR if it isn't showing any activity.

## Community

* Join our community forum
* Follow us on Twitter
* Participate in discussions on GitHub issues

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Something isn't working
* `enhancement`: New feature or request
* `documentation`: Improvements or additions to documentation
* `good first issue`: Good for newcomers
* `help wanted`: Extra attention is needed

Thank you for contributing to OpenLLM! 