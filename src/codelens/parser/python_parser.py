import ast
from pathlib import Path

from .models import Class, Function, Import


class PythonAstVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.classes: list[Class] = []
        self.functions: list[Function] = []
        self.imports: list[Import] = []  # Store imported dependencies
        self.current_class: Class | None = None  # Pointer to the current class (for methods)
        self.current_function: Function | None = None  # Pointer to the current function

    def visit_ClassDef(self, node: ast.ClassDef):
        # Extract base class names (from which the current class inherits)
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
            else:
                bases.append("UnknownBase")

        cls_symbol = Class(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            bases=bases,
            end_line_number=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
        )
        self.classes.append(cls_symbol)

        # Save the pointer so that the following functions are written as methods of this class
        previous_class = self.current_class
        self.current_class = cls_symbol
        # Continue traversal inside the class (parse methods)
        self.generic_visit(node)

        # RESTORE CONTEXT when exiting the class
        self.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # Collect function argument names
        args = [arg.arg for arg in node.args.args]

        func = Function(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            args=args,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            end_line_number=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
        )

        # Inside a class the function is a method; otherwise it is a global function
        if self.current_class:
            self.current_class.methods.append(func)
        else:
            self.functions.append(func)

        # SAVE CONTEXT before diving inside the function
        previous_function = self.current_function
        self.current_function = func

        self.generic_visit(node)

        # RESTORE CONTEXT after exiting
        self.current_function = previous_function

    # Support for async functions (async def)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            # Store a tuple (name, call line number)
            if isinstance(node.func, ast.Name):
                self.current_function.calls.append((node.func.id, node.lineno))
            elif isinstance(node.func, ast.Attribute):
                self.current_function.calls.append((node.func.attr, node.lineno))

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(
                Import(file_path=self.file_path, module=None, name=alias.name, alias=alias.asname)
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module if node.module else ""
        for alias in node.names:
            self.imports.append(
                Import(
                    file_path=self.file_path,
                    module=module_name,
                    name=alias.name,
                    alias=alias.asname,
                )
            )
        self.generic_visit(node)


def parse_python_file(
    path: Path, record_as: str | None = None
) -> tuple[list[Class], list[Function], list[Import]]:
    """Reads the file, builds an AST and returns the found classes, functions, and imports.

    `record_as` is the path stored on every returned symbol. The indexer passes
    the repository-relative path so callers never have to rewrite `file_path`
    after the fact; it defaults to the path that was read.
    """
    try:
        code = path.read_text(encoding="utf-8")
    except Exception:
        return [], [], []

    # Catch SyntaxError so broken files don't stop the whole indexer
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], [], []

    visitor = PythonAstVisitor(record_as if record_as is not None else str(path))
    visitor.visit(tree)

    return visitor.classes, visitor.functions, visitor.imports
