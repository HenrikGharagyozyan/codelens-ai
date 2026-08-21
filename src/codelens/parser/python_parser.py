import ast
from pathlib import Path
from .models import Class, Function


class PythonAstVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.classes: list[Class] = []
        self.functions: list[Function] = []
        self.current_class: Class | None = None  # Pointer to the current class (for methods)
        self.current_function: Function | None = None  # Pointer to the current function


    def visit_ClassDef(self, node: ast.ClassDef):
        # Extract base class names (from which the current class inherits)
        bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
        
        cls_symbol = Class(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            bases=bases,
            end_line_number=getattr(node, 'end_lineno', node.lineno)
        )
        self.classes.append(cls_symbol)
        
        # Save the pointer so that the following functions are written as methods of this class
        self.current_class = cls_symbol
        # Continue traversal inside the class (parse methods)
        self.generic_visit(node)
        # Remove the pointer when exiting the class
        self.current_class = None


    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # Collect function argument names
        args = [arg.arg for arg in node.args.args]
        
        func = Function(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            args=args,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            end_line_number=getattr(node, 'end_lineno', node.lineno)
        )
        
        # If we are currently inside a class, add the function to methods, otherwise to global functions
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
            # If it's a simple call (e.g. print(), my_func())
            if isinstance(node.func, ast.Name):
                self.current_function.calls.append(node.func.id)
            # If it's a method call (e.g. self.scan(), db.insert())
            elif isinstance(node.func, ast.Attribute):
                self.current_function.calls.append(node.func.attr)
                
        self.generic_visit(node)


def parse_python_file(path: Path) -> tuple[list[Class], list[Function]]:
    """Reads the file, builds an AST and returns the found classes and functions."""
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    
    visitor = PythonAstVisitor(str(path))
    visitor.visit(tree)
    
    return visitor.classes, visitor.functions