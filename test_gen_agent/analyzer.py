"""AST-based code analyzer for extracting functions, classes, type hints, and docstrings."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FunctionInfo:
    """Information about a single function or method."""

    name: str
    qualname: str  # dotted name, e.g. MyClass.my_method
    args: list[str]
    arg_types: dict[str, str]  # arg name -> annotation string
    return_type: str | None
    docstring: str | None
    is_method: bool
    is_classmethod: bool
    is_staticmethod: bool
    is_async: bool
    lineno: int
    source: str  # full source of the function
    raises: list[str]  # exception types mentioned in the docstring

    def signature(self) -> str:
        """Return a human-readable function signature."""
        parts: list[str] = []
        for arg in self.args:
            ann = self.arg_types.get(arg)
            parts.append(f"{arg}: {ann}" if ann else arg)
        ret = f" -> {self.return_type}" if self.return_type else ""
        prefix = "async def" if self.is_async else "def"
        return f"{prefix} {self.name}({', '.join(parts)}){ret}"


@dataclass
class ClassInfo:
    """Information about a single class."""

    name: str
    bases: list[str]
    docstring: str | None
    methods: list[FunctionInfo] = field(default_factory=list)
    lineno: int = 0


@dataclass
class ModuleInfo:
    """All extracted information from a Python source file."""

    path: Path
    module_name: str
    docstring: str | None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def all_functions(self) -> list[FunctionInfo]:
        """Return module-level functions and all class methods."""
        result = list(self.functions)
        for cls in self.classes:
            result.extend(cls.methods)
        return result


class CodeAnalyzer:
    """Parse Python source files and extract structural information."""

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def analyze_file(self, path: str | Path) -> ModuleInfo:
        """Analyze a Python source file and return a ModuleInfo."""
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        return self._analyze_source(source, path)

    def analyze_source(self, source: str, path: str | Path = Path("<string>")) -> ModuleInfo:
        """Analyze a string of Python source code and return a ModuleInfo."""
        return self._analyze_source(source, Path(path))

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _analyze_source(self, source: str, path: Path) -> ModuleInfo:
        tree = ast.parse(source, filename=str(path))
        module_name = path.stem

        module = ModuleInfo(
            path=path,
            module_name=module_name,
            docstring=ast.get_docstring(tree),
        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module.imports.append(ast.unparse(node))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self._extract_function(node, source, qualname=node.name)
                module.functions.append(fn)
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node, source)
                module.classes.append(cls)

        return module

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        qualname: str = "",
        is_method: bool = False,
        is_classmethod: bool = False,
        is_staticmethod: bool = False,
    ) -> FunctionInfo:
        args = [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        arg_types: dict[str, str] = {}
        for a in node.args.args:
            if a.annotation:
                arg_types[a.arg] = ast.unparse(a.annotation)

        return_type: str | None = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        docstring = ast.get_docstring(node)
        raises = self._extract_raises(docstring) if docstring else []

        src_lines = source.splitlines()
        fn_lines = src_lines[node.lineno - 1 : node.end_lineno]
        fn_source = textwrap.dedent("\n".join(fn_lines))

        return FunctionInfo(
            name=node.name,
            qualname=qualname or node.name,
            args=args,
            arg_types=arg_types,
            return_type=return_type,
            docstring=docstring,
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            lineno=node.lineno,
            source=fn_source,
            raises=raises,
        )

    def _extract_class(self, node: ast.ClassDef, source: str) -> ClassInfo:
        bases = [ast.unparse(b) for b in node.bases]
        cls = ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            lineno=node.lineno,
        )

        for item in node.body:
            is_classmethod = False
            is_staticmethod = False

            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in item.decorator_list:
                    dec_str = ast.unparse(decorator)
                    if dec_str == "classmethod":
                        is_classmethod = True
                    elif dec_str == "staticmethod":
                        is_staticmethod = True

                fn = self._extract_function(
                    item,
                    source,
                    qualname=f"{node.name}.{item.name}",
                    is_method=True,
                    is_classmethod=is_classmethod,
                    is_staticmethod=is_staticmethod,
                )
                cls.methods.append(fn)

        return cls

    @staticmethod
    def _extract_raises(docstring: str) -> list[str]:
        """Parse exception types from a docstring (Google/NumPy/reST styles)."""
        exceptions: list[str] = []
        for line in docstring.splitlines():
            stripped = line.strip()
            # reST: `:raises ValueError:`
            if stripped.startswith(":raises") or stripped.startswith(":raise"):
                parts = stripped.split()
                if len(parts) >= 2:
                    exc = parts[1].rstrip(":")
                    if exc:
                        exceptions.append(exc)
            # Google: `    ValueError: ...`
            elif ":" in stripped and not stripped.startswith(">>>"):
                candidate = stripped.split(":")[0].strip()
                # heuristic: PascalCase or ends with Error/Exception/Warning
                if candidate and (
                    candidate[0].isupper()
                    and all(c.isalnum() or c == "_" for c in candidate)
                    and (
                        candidate.endswith("Error")
                        or candidate.endswith("Exception")
                        or candidate.endswith("Warning")
                    )
                ):
                    exceptions.append(candidate)
        return exceptions
