import ast
from dataclasses import dataclass, field
from pathlib import Path

from .models import UNKNOWN_RECEIVER, CallSite, Class, Function, Import, Module, Variable


def dotted_path(node: ast.expr) -> str | None:
    """Renders a name chain as `a.b.c`, with `()` marking the result of a call.

    `self.db` -> "self.db", `Service().run` -> "Service().run",
    `super()` -> "super()". Anything that is not a plain chain (subscripts,
    literals, lambdas) gives None, since nothing downstream could follow it.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_path(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Call):
        base = dotted_path(node.func)
        return f"{base}()" if base else None
    return None


# Literals whose methods are builtins: `", ".join(parts)` or `lines.append(x)`.
LITERAL_TYPES: dict[type, str] = {
    ast.List: "list()",
    ast.ListComp: "list()",
    ast.Dict: "dict()",
    ast.DictComp: "dict()",
    ast.Set: "set()",
    ast.SetComp: "set()",
    ast.Tuple: "tuple()",
    ast.JoinedStr: "str()",
}

# Builtin containers named in annotations, bare or subscripted: `list[str]`.
BUILTIN_CONTAINERS = {
    "list": "list",
    "dict": "dict",
    "set": "set",
    "frozenset": "frozenset",
    "tuple": "tuple",
    "typing.List": "list",
    "typing.Dict": "dict",
    "typing.Set": "set",
    "typing.Tuple": "tuple",
    "List": "list",
    "Dict": "dict",
    "Set": "set",
    "Tuple": "tuple",
}


def literal_type(node: ast.expr) -> str | None:
    """The builtin type of a literal, as an instance path: `[]` -> "list()", `"x"` -> "str()"."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "str()"
        if isinstance(node.value, bytes):
            return "bytes()"
        return None
    return LITERAL_TYPES.get(type(node))


def annotation_type(node: ast.expr | None) -> str | None:
    """Reduces a type annotation to the single class a resolver can use.

    `Foo`, `mod.Foo`, `Optional[Foo]`, `Foo | None` and `"Foo"` all give "Foo".
    A builtin container such as `list[Foo]` gives "list": a method called on
    the list is a list method, not a method of Foo. Other generics give None.
    """
    if node is None:
        return None

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return annotation_type(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        options = [t for t in (annotation_type(node.left), annotation_type(node.right)) if t]
        return options[0] if len(options) == 1 else None

    if isinstance(node, ast.Subscript):
        outer = dotted_path(node.value)
        if outer in ("Optional", "typing.Optional"):
            return annotation_type(node.slice)
        return BUILTIN_CONTAINERS.get(outer or "")

    if isinstance(node, (ast.Name, ast.Attribute)):
        path = dotted_path(node)
        return None if path == "None" else path

    return None


def value_type(node: ast.expr | None, local_types: dict[str, str | None]) -> str | None:
    """Describes what an expression evaluates to, as a receiver-style path.

    `Foo(...)` gives "Foo()" (an instance), `self.db` gives "self.db". A leading
    local name is replaced by what is known about it, so the result still means
    something outside the function it was written in: with `db: Database` as a
    parameter, `db if db else Database()` gives "Database".
    """
    if node is None:
        return None

    if isinstance(node, ast.IfExp):
        return value_type(node.body, local_types) or value_type(node.orelse, local_types)

    if isinstance(node, ast.BoolOp):
        # `db or Database()`: the first operand that says anything wins.
        for operand in node.values:
            described = value_type(operand, local_types)
            if described:
                return described
        return None

    if isinstance(node, ast.Await):
        return value_type(node.value, local_types)

    path = dotted_path(node)
    if path is None:
        return literal_type(node)

    head, dot, rest = path.partition(".")
    if head in local_types:
        known = local_types[head]
        return f"{known}{dot}{rest}" if known else None

    return path


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def class_signature(node: ast.ClassDef) -> str:
    parents = [ast.unparse(base) for base in node.bases] + [ast.unparse(kw) for kw in node.keywords]
    return f"class {node.name}({', '.join(parents)})" if parents else f"class {node.name}"


def argument_names(args: ast.arguments) -> list[str]:
    """Every parameter in declaration order, with `*args` and `**kwargs` marked."""
    names = [a.arg for a in args.posonlyargs + args.args]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return names


def base_name(node: ast.expr) -> str:
    """The short display name of a base class expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "UnknownBase"


class PythonAstVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.classes: list[Class] = []
        self.functions: list[Function] = []
        self.variables: list[Variable] = []
        self.imports: list[Import] = []  # Store imported dependencies
        self.current_class: Class | None = None  # Pointer to the current class (for methods)
        self.current_function: Function | None = None  # Pointer to the current function
        # How many local functions or classes we are inside. Their bodies belong
        # to the enclosing function, but their `return` statements do not.
        self.local_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        if self.current_function is not None:
            # A class defined inside a function is local to it, just like a
            # nested helper function: not a symbol, and its calls belong to the
            # enclosing function.
            self._visit_local_definition(node)
            return

        # Bases, keywords and decorators are evaluated in the enclosing scope,
        # not inside the class.
        for expr in [*node.decorator_list, *node.bases, *(kw.value for kw in node.keywords)]:
            self.visit(expr)

        parent = self.current_class
        cls_symbol = Class(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
            qualname=f"{parent.qualname}.{node.name}" if parent else node.name,
            signature=class_signature(node),
            decorators=[ast.unparse(d) for d in node.decorator_list],
            bases=[base_name(base) for base in node.bases],
            base_refs=[dotted_path(base) for base in node.bases],
        )
        self.classes.append(cls_symbol)

        # Save the pointer so that the following functions are written as methods of this class
        self.current_class = cls_symbol
        for statement in node.body:
            self.visit(statement)

        # RESTORE CONTEXT when exiting the class
        self.current_class = parent

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # Check if we are already inside a function (prevents nested helpers from escaping)
        if self.current_function is not None:
            # We are inside a nested helper function.
            # Do not create a separate symbol. Calls inside will be attributed to the parent function.
            self._visit_local_definition(node)
            return

        # Decorators, default values and annotations run in the enclosing scope,
        # so a call such as `@app.command()` is not a call made by this function.
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

        cls = self.current_class
        func = Function(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", node.lineno),
            docstring=ast.get_docstring(node),
            qualname=f"{cls.qualname}.{node.name}" if cls else node.name,
            signature=function_signature(node),
            decorators=[ast.unparse(d) for d in node.decorator_list],
            args=argument_names(node.args),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            returns=ast.unparse(node.returns) if node.returns is not None else None,
            local_types=self._parameter_types(node.args, is_method=cls is not None),
        )

        if cls:
            cls.methods.append(func)
        else:
            self.functions.append(func)

        # SAVE CONTEXT before diving inside the function
        previous_function = self.current_function
        self.current_function = func

        for statement in node.body:
            self.visit(statement)

        # RESTORE CONTEXT after exiting
        self.current_function = previous_function

        # An explicit annotation beats whatever the body happened to return.
        if node.returns is not None:
            func.return_type = annotation_type(node.returns)

    # Support for async functions (async def)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def _visit_local_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        # The local name shadows any global of the same name inside this function.
        self._declare_local(node.name)
        self.local_depth += 1
        self.generic_visit(node)
        self.local_depth -= 1

    def _declare_local(self, target: ast.expr | str) -> None:
        """Marks names bound inside the current function as local, of unknown type.

        Without this, `for db in shards: db.query()` would resolve `db` as the
        module-level name `db`, which it is not.
        """
        func = self.current_function
        if func is None:
            return
        if isinstance(target, str):
            func.local_types.setdefault(target, None)
        elif isinstance(target, ast.Name):
            func.local_types.setdefault(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._declare_local(element)
        elif isinstance(target, ast.Starred):
            self._declare_local(target.value)

    def visit_For(self, node: ast.For | ast.AsyncFor):
        self._declare_local(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit_For(node)

    def visit_With(self, node: ast.With | ast.AsyncWith):
        for item in node.items:
            if item.optional_vars is not None:
                self._declare_local(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            self._declare_local(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        self._declare_local(node.target)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda):
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            self._declare_local(arg.arg)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        func = self.current_function
        if func is not None and func.local_types.get(node.target.id) is None:
            func.local_types[node.target.id] = value_type(node.value, func.local_types)
        self.generic_visit(node)

    @staticmethod
    def _parameter_types(args: ast.arguments, is_method: bool) -> dict[str, str | None]:
        """Parameter annotations, keyed by name; unannotated parameters map to None."""
        params = args.posonlyargs + args.args
        if is_method and params and params[0].arg in ("self", "cls"):
            # `self` and `cls` are resolved from the enclosing class instead.
            params = params[1:]

        types: dict[str, str | None] = {p.arg: annotation_type(p.annotation) for p in params + args.kwonlyargs}
        for starred in (args.vararg, args.kwarg):
            if starred is not None:
                types[starred.arg] = None
        return types

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            func = node.func
            if isinstance(func, ast.Name):
                self.current_function.calls.append(CallSite(func.id, node.lineno))
            elif isinstance(func, ast.Attribute):
                receiver = dotted_path(func.value) or literal_type(func.value) or UNKNOWN_RECEIVER
                self.current_function.calls.append(CallSite(func.attr, node.lineno, receiver))

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        func = self.current_function
        if func is not None and self.local_depth == 0 and func.return_type is None:
            func.return_type = value_type(node.value, func.local_types)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(Import(file_path=self.file_path, module=None, name=alias.name, alias=alias.asname))
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
                    level=node.level or 0,
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._record_assignment(target, node, node.value, annotation=None)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # For annotated variables (for example, CONST: str = "...")
        self._record_assignment(node.target, node, node.value, annotation=node.annotation)
        self.generic_visit(node)

    def _record_assignment(self, target: ast.expr, node: ast.stmt, value: ast.expr | None, annotation):
        """Records what an assignment tells us, depending on where it happens.

        - module level: a global variable symbol;
        - class body: the type of a class attribute;
        - function body: the type of a local name, or of `self.<attr>`.
        """
        func = self.current_function

        if func is not None:
            described = annotation_type(annotation) or value_type(value, func.local_types)
            if isinstance(target, ast.Name):
                # First informative assignment wins; `x = None` never hides a
                # later `x = Foo()`.
                if func.local_types.get(target.id) is None:
                    func.local_types[target.id] = described
            elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
                self._declare_local(target)
            elif (
                self.current_class is not None
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and described
            ):
                self.current_class.attr_types.setdefault(target.attr, described)
            return

        if not isinstance(target, ast.Name):
            return

        described = annotation_type(annotation) or value_type(value, {})

        if self.current_class is not None:
            if described:
                self.current_class.attr_types.setdefault(target.id, described)
            return

        # Capture only global variables (not inside classes or functions)
        self.variables.append(
            Variable(
                name=target.id,
                file_path=self.file_path,
                line_number=node.lineno,
                end_line_number=getattr(node, "end_lineno", node.lineno),
                signature=ast.unparse(node).splitlines()[0][:200],
                value_type=described,
            )
        )


@dataclass
class ParsedFile:
    """Everything one file yielded: its module summary and its symbols."""

    module: Module | None = None
    classes: list[Class] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    variables: list[Variable] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)


def parse_file(path: Path, record_as: str | None = None) -> ParsedFile:
    """Parses one Python file into a `ParsedFile`.

    `record_as` is the path stored on every returned symbol. The indexer passes
    the repository-relative path so callers never have to rewrite `file_path`
    after the fact; it defaults to the path that was read.
    """
    try:
        code = path.read_text(encoding="utf-8")
    except Exception:
        return ParsedFile()

    # Catch SyntaxError so broken files don't stop the whole indexer
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ParsedFile()

    file_path = record_as if record_as is not None else str(path)
    visitor = PythonAstVisitor(file_path)
    visitor.visit(tree)

    top_level = sorted(visitor.functions + visitor.variables, key=lambda sym: sym.line_number)
    module = Module(
        name=Path(file_path).stem,
        file_path=file_path,
        line_number=1,
        end_line_number=len(code.splitlines()) or 1,
        docstring=ast.get_docstring(tree),
        top_level_names=[cls.name for cls in visitor.classes] + [sym.name for sym in top_level],
    )

    return ParsedFile(
        module=module,
        classes=visitor.classes,
        functions=visitor.functions,
        variables=visitor.variables,
        imports=visitor.imports,
    )


def parse_python_file(path: Path, record_as: str | None = None) -> tuple[list[Class], list[Function], list[Import]]:
    """Reads the file, builds an AST and returns the found classes, functions, and imports.

    Kept as the symbol-only view over `parse_file`, which callers that do not
    need the module summary can keep using.
    """
    parsed = parse_file(path, record_as)
    return parsed.classes, parsed.functions, parsed.imports
