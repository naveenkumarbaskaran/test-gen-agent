# test-gen-agent

An AI-powered pytest test suite generator that uses Claude (`claude-sonnet-4-6`) to read your Python source files and automatically produce comprehensive test suites covering happy-path, edge-case, and error scenarios.

## Features

- Parses Python source files with the built-in `ast` module — no external parser required.
- Extracts functions, methods, classes, type hints, and docstrings.
- Uses an agentic loop with three introspection tools so Claude can fully understand your code before writing tests.
- Generates `pytest`-style tests with `assert`, `pytest.raises`, and `pytest.mark.parametrize`.
- CLI with progress spinner, syntax-highlighted preview, and configurable output directory.

## Installation

```bash
pip install test-gen-agent
# or from source:
pip install -e .
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Generate tests for a single file

```bash
test-gen generate src/mymodule.py --output tests/
```

### Generate tests for an entire directory

```bash
test-gen generate src/ --output tests/ --coverage-target 90
```

### Preview output without writing files

```bash
test-gen generate src/mymodule.py --print
```

### All options

```
Usage: test-gen generate [OPTIONS] SOURCE

  Generate pytest tests for SOURCE (a file or directory).

Arguments:
  SOURCE  Python file or directory to generate tests for.  [required]

Options:
  --framework [pytest]    Test framework to use.  [default: pytest]
  -o, --output PATH       Output directory for generated test files.  [default: tests/]
  --coverage-target INT   Minimum coverage percentage to aim for.  [default: 80]
  --print                 Print generated tests to stdout instead of writing files.
  --api-key TEXT          Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
  --help                  Show this message and exit.
```

## Python API

You can also use the agent programmatically:

```python
from test_gen_agent.agent import TestGenAgent

agent = TestGenAgent()  # reads ANTHROPIC_API_KEY from environment
test_code = agent.generate_tests(
    source_path="src/calculator.py",
    framework="pytest",
    coverage_target=85,
)
print(test_code)
```

## How it works

The agent exposes three tools to Claude during an agentic loop:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read the raw source text of a Python file |
| `list_functions(path)` | Return all function/method signatures, docstrings, and raised exceptions |
| `get_type_hints(path, func)` | Return argument and return-type annotations for a specific function |

Claude calls these tools to understand the code structure, then generates tests in a single final response.  The `CodeAnalyzer` class uses Python's built-in `ast` module to extract all structural information without executing the source file.

## Project structure

```
test_gen_agent/
    __init__.py      Package init
    agent.py         TestGenAgent — agentic loop + tool implementations
    analyzer.py      CodeAnalyzer — AST-based source introspection
    cli.py           Click CLI with Rich output
pyproject.toml
README.md
```

## Requirements

- Python 3.11+
- `anthropic` >= 0.40.0
- `click` >= 8.1
- `rich` >= 13.0

## License

MIT
