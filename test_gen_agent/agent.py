"""TestGenAgent: uses Claude to generate pytest test suites from Python source."""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Any

import anthropic

from .analyzer import CodeAnalyzer, FunctionInfo, ModuleInfo


MODEL = "claude-sonnet-4-6"


class TestGenAgent:
    """AI agent that reads Python source files and generates pytest test suites.

    The agent exposes three tools to Claude:
      - read_file       : read raw source text of a file
      - list_functions  : return function signatures in a file
      - get_type_hints  : return argument and return type annotations for a function

    Claude uses those tools to understand the code and then writes a pytest
    test module covering happy-path, edge, and error cases.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY by default
        self._analyzer = CodeAnalyzer()
        self._file_cache: dict[str, str] = {}
        self._module_cache: dict[str, ModuleInfo] = {}

        self._tools: list[dict[str, Any]] = [
            {
                "name": "read_file",
                "description": (
                    "Read the raw source text of a Python source file. "
                    "Returns the full file contents as a string."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the Python source file.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "list_functions",
                "description": (
                    "Parse a Python source file with the AST and return a JSON list of function "
                    "signatures (including methods). Each entry contains: name, qualname, "
                    "signature, is_async, lineno, docstring, and raises."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the Python source file.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "get_type_hints",
                "description": (
                    "Return a JSON object with argument names mapped to their type annotations "
                    "and the return type for a specific function or method in a file. "
                    "Use the dotted qualname (e.g. 'MyClass.my_method') to identify a method."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path to the Python source file.",
                        },
                        "func": {
                            "type": "string",
                            "description": (
                                "The simple name or dotted qualname of the function, "
                                "e.g. 'add', 'MyClass.add'."
                            ),
                        },
                    },
                    "required": ["path", "func"],
                },
            },
        ]

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def generate_tests(
        self,
        source_path: str | Path,
        framework: str = "pytest",
        coverage_target: int = 80,
    ) -> str:
        """Generate a pytest test module for *source_path* and return it as a string.

        Args:
            source_path: Path to the Python file to test.
            framework: Test framework to use (currently only 'pytest' is supported).
            coverage_target: Minimum line-coverage percentage the tests should aim for.

        Returns:
            A string containing valid Python test code.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        system_prompt = self._build_system_prompt(framework, coverage_target)
        user_message = (
            f"Generate a comprehensive {framework} test suite for the file: `{source_path}`.\n"
            f"Aim for at least {coverage_target}% line coverage.\n"
            "Use the available tools to inspect the file before writing tests.\n"
            "Return ONLY the complete Python test file content—no markdown, no explanation."
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        # Agentic loop: keep going until Claude stops calling tools
        while True:
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=system_prompt,
                tools=self._tools,
                messages=messages,
            )

            # Append Claude's full response to message history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract text from the final response
                for block in response.content:
                    if hasattr(block, "text") and block.type == "text":
                        return self._clean_code_block(block.text)
                return ""

            if response.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._dispatch_tool(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — return whatever text we have
            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    return self._clean_code_block(block.text)
            return ""

    # ---------------------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------------------

    def _dispatch_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Route a tool call to the correct implementation and return a JSON string."""
        try:
            if name == "read_file":
                return self._tool_read_file(**tool_input)
            elif name == "list_functions":
                return self._tool_list_functions(**tool_input)
            elif name == "get_type_hints":
                return self._tool_get_type_hints(**tool_input)
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})

    def _tool_read_file(self, path: str) -> str:
        """Return the raw source text of a file."""
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        source = p.read_text(encoding="utf-8")
        self._file_cache[str(p.resolve())] = source
        return source

    def _tool_list_functions(self, path: str) -> str:
        """Return a JSON list of function signatures in the file."""
        module = self._get_module(path)
        result = []
        for fn in module.all_functions():
            result.append(
                {
                    "name": fn.name,
                    "qualname": fn.qualname,
                    "signature": fn.signature(),
                    "is_async": fn.is_async,
                    "lineno": fn.lineno,
                    "docstring": fn.docstring,
                    "raises": fn.raises,
                }
            )
        return json.dumps(result, indent=2)

    def _tool_get_type_hints(self, path: str, func: str) -> str:
        """Return arg/return type annotations for a specific function."""
        module = self._get_module(path)
        # Search both module-level and class methods
        for fn in module.all_functions():
            if fn.qualname == func or fn.name == func:
                return json.dumps(
                    {
                        "qualname": fn.qualname,
                        "arg_types": fn.arg_types,
                        "return_type": fn.return_type,
                    },
                    indent=2,
                )
        return json.dumps({"error": f"Function '{func}' not found in {path}"})

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _get_module(self, path: str) -> ModuleInfo:
        key = str(Path(path).resolve())
        if key not in self._module_cache:
            self._module_cache[key] = self._analyzer.analyze_file(path)
        return self._module_cache[key]

    @staticmethod
    def _build_system_prompt(framework: str, coverage_target: int) -> str:
        return f"""\
You are an expert Python test engineer. Your job is to write high-quality \
{framework} test suites for the Python source files a user provides.

Guidelines:
- Use the tools to thoroughly understand the code before writing tests.
- Always call list_functions first, then read_file and get_type_hints as needed.
- Write tests covering:
  1. Happy path: normal inputs that should work correctly.
  2. Edge cases: empty inputs, zero values, boundary conditions, large inputs.
  3. Error cases: invalid types, values that should raise exceptions.
- Aim for at least {coverage_target}% line coverage.
- Use pytest conventions: `def test_<name>():` functions, `assert` statements.
- Use `pytest.raises` for expected exceptions.
- Use `pytest.mark.parametrize` for data-driven tests where appropriate.
- Include module-level imports and any required fixtures.
- Do NOT include markdown code fences in your output.
- Return ONLY valid Python source code for the test file.
"""

    @staticmethod
    def _clean_code_block(text: str) -> str:
        """Strip markdown code fences if Claude accidentally included them."""
        lines = text.strip().splitlines()
        # Remove leading ```python or ``` fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove trailing ``` fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip() + "\n"
