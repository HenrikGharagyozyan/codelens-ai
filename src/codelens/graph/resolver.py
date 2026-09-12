"""Resolving call sites to the symbols they actually call.

The parser records every call as a name plus the expression it was made on
(`connect`, `self.db.search`, `Service().run`). Matching on the name alone
links a call to `connect()` with every `connect` in the repository. This module
follows Python's own lookup rules, as far as they can be followed statically:

1. a bare name is looked up in the file: its own definitions, then its imports;
2. `self.x()` and `cls.x()` are looked up on the enclosing class and its bases;
3. `super().x()` is looked up on the bases only;
4. `a.b.x()` walks the chain: modules through imports, and objects through the
   type the code gives them (parameter annotations, `x = Foo()`,
   `self.x = Foo()`, class annotations, property and function return types).

What cannot be decided statically stays unresolved rather than guessed. A wrong
edge is worse than a missing one: the LLM treats the graph as fact.
"""

import builtins
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath

from codelens.parser.models import UNKNOWN_RECEIVER, CallSite, Class, Function, Variable
from codelens.parser.python_parser import ParsedFile

# Bounds every recursive lookup, so a cyclic import or a self-referencing
# attribute type ends in "unresolved" instead of a RecursionError.
MAX_DEPTH = 12

# How a call was resolved, or why it was not. Stored in `calls.resolution`.
DIRECT = "direct"  # foo() defined in the same file
IMPORT = "import"  # foo() bound by an import
SELF = "self"  # self.foo() / cls.foo(), including inherited methods
SUPER = "super"  # super().foo()
MODULE = "module"  # module.foo() through an imported module
TYPED = "typed"  # obj.foo() where obj's class is known
BUILTIN = "builtin"  # len(), str.join() and friends
EXTERNAL = "external"  # third-party or standard-library code
UNRESOLVED = "unresolved"  # inside the repository, perhaps, but not decidable

RESOLVED = frozenset({DIRECT, IMPORT, SELF, SUPER, MODULE, TYPED})

CALLABLE_KINDS = frozenset({"function", "method", "class"})
PROPERTY_DECORATORS = frozenset({"property", "cached_property", "functools.cached_property"})
BUILTIN_NAMES = frozenset(dir(builtins))


@dataclass(frozen=True)
class ModuleRef:
    """A module or package inside the repository, by dotted name."""

    name: str


@dataclass(frozen=True)
class SymbolRef:
    """A symbol inside the repository. A class also stands for its instances."""

    symbol_id: str
    kind: str  # "class" | "function" | "method" | "variable"


@dataclass(frozen=True)
class Opaque:
    """A value known to come from outside the repository."""

    resolution: str  # BUILTIN or EXTERNAL


EXTERNAL_VALUE = Opaque(EXTERNAL)
BUILTIN_VALUE = Opaque(BUILTIN)

Target = ModuleRef | SymbolRef | Opaque


@dataclass(frozen=True)
class Context:
    """Where an expression is evaluated: the enclosing class and the local names."""

    class_id: str | None
    local_types: dict[str, str | None]


NO_LOCALS: dict[str, str | None] = {}


def module_name(path: str) -> str:
    """`src/codelens/db.py` -> "src.codelens.db"; a package's `__init__.py` names the package."""
    parts = list(PurePath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class ModuleRegistry:
    """Maps dotted module names to repository files.

    A file is registered under every suffix of its dotted path, because the
    repository root is rarely the import root: `src/codelens/db.py` is imported
    as `codelens.db`, and a test may import it as `db`. A suffix shared by two
    files is ambiguous and resolves to nothing rather than to a guess.
    """

    def __init__(self, paths):
        self.full_names: dict[str, str] = {}
        self._files: dict[str, set[str]] = defaultdict(set)
        self._packages: set[str] = set()

        for path in paths:
            full = module_name(path)
            self.full_names[path] = full
            parts = full.split(".") if full else []
            for start in range(len(parts)):
                suffix = parts[start:]
                self._files[".".join(suffix)].add(path)
                for end in range(1, len(suffix)):
                    self._packages.add(".".join(suffix[:end]))

    def file_for(self, dotted: str) -> str | None:
        files = self._files.get(dotted)
        return next(iter(files)) if files and len(files) == 1 else None

    def is_known(self, dotted: str) -> bool:
        """True for any module or package that lives in the repository."""
        return dotted in self._files or dotted in self._packages


class SymbolTable:
    """Everything the resolver knows about a parsed repository.

    Built once per indexing run from every `ParsedFile`, after the indexer has
    assigned `symbol_id` to each symbol.
    """

    def __init__(self, files: dict[str, ParsedFile]):
        self.modules = ModuleRegistry(files)

        self._definitions: dict[str, dict[str, SymbolRef]] = {}
        self._bindings: dict[str, dict[str, tuple[str, str, str]]] = {}
        self._stars: dict[str, list[str]] = {}

        self._classes: dict[str, Class] = {}
        self._methods: dict[str, dict[str, str]] = {}
        self._nested: dict[str, dict[str, str]] = defaultdict(dict)
        self._functions: dict[str, Function] = {}
        self._function_class: dict[str, str | None] = {}
        self._variables: dict[str, Variable] = {}
        self._file_of: dict[str, str] = {}

        self._base_cache: dict[str, list[str | None]] = {}
        self._mro_cache: dict[str, list[str]] = {}

        for path, parsed in files.items():
            self._register(path, parsed)

    # ------------------------------------------------------------ registration

    def _register(self, path: str, parsed: ParsedFile) -> None:
        definitions: dict[str, SymbolRef] = {}
        class_by_qualname: dict[str, str] = {}

        for cls in parsed.classes:
            if cls.symbol_id is None:
                continue
            class_id = cls.symbol_id
            self._classes[class_id] = cls
            self._file_of[class_id] = path
            class_by_qualname.setdefault(cls.qualname, class_id)

            parent_qualname, _, short_name = cls.qualname.rpartition(".")
            if parent_qualname:
                parent_id = class_by_qualname.get(parent_qualname)
                if parent_id:
                    self._nested[parent_id].setdefault(short_name, class_id)
            else:
                definitions.setdefault(cls.name, SymbolRef(class_id, "class"))

            methods: dict[str, str] = {}
            for method in cls.methods:
                if method.symbol_id is None:
                    continue
                # First definition wins: a property getter before its setter.
                methods.setdefault(method.name, method.symbol_id)
                self._register_function(method, path, class_id)
            self._methods[class_id] = methods

        for func in parsed.functions:
            if func.symbol_id is not None:
                definitions.setdefault(func.name, SymbolRef(func.symbol_id, "function"))
                self._register_function(func, path, None)

        for var in parsed.variables:
            if var.symbol_id is not None:
                definitions.setdefault(var.name, SymbolRef(var.symbol_id, "variable"))
                self._variables[var.symbol_id] = var
                self._file_of[var.symbol_id] = path

        self._definitions[path] = definitions
        self._bindings[path], self._stars[path] = self._import_bindings(path, parsed)

    def _register_function(self, func: Function, path: str, class_id: str | None) -> None:
        assert func.symbol_id is not None
        self._functions[func.symbol_id] = func
        self._function_class[func.symbol_id] = class_id
        self._file_of[func.symbol_id] = path

    def _import_bindings(self, path: str, parsed: ParsedFile):
        """The names each import statement binds in the file's namespace."""
        bindings: dict[str, tuple[str, str, str]] = {}
        stars: list[str] = []

        for imp in parsed.imports:
            if imp.module is None:
                # `import a.b.c` binds `a`; `import a.b.c as x` binds `x` to a.b.c.
                if imp.alias:
                    bindings.setdefault(imp.alias, ("module", imp.name, ""))
                else:
                    head = imp.name.split(".")[0]
                    bindings.setdefault(head, ("module", head, ""))
                continue

            absolute = self._absolute_module(path, imp.module, imp.level)
            if imp.name == "*":
                stars.append(absolute)
            else:
                bindings.setdefault(imp.alias or imp.name, ("from", absolute, imp.name))

        return bindings, stars

    def _absolute_module(self, path: str, module: str, level: int) -> str:
        if level == 0:
            return module
        package = self.modules.full_names[path].split(".")
        if not path.endswith("__init__.py"):
            package = package[:-1]  # a module's package is its directory
        # Level 1 is the current package; every extra dot goes one level up.
        if level > 1:
            package = package[: max(0, len(package) - (level - 1))]
        return ".".join(part for part in (".".join(package), module) if part)

    # ------------------------------------------------------------- public API

    def resolve_call(self, function_id: str, call: CallSite) -> tuple[str | None, str]:
        """Resolves one call made inside `function_id`.

        Returns (callee symbol id or None, resolution label).
        """
        func = self._functions.get(function_id)
        if func is None:
            return None, UNRESOLVED

        path = self._file_of[function_id]
        ctx = Context(self._function_class[function_id], func.local_types)

        if call.receiver is None:
            return self._resolve_bare_call(path, ctx, call.name)

        if call.receiver == UNKNOWN_RECEIVER:
            return None, UNRESOLVED

        if call.receiver == "super()":
            if ctx.class_id is None:
                return None, UNRESOLVED
            method_id = self.find_method(ctx.class_id, call.name, skip_own=True)
            if method_id:
                return method_id, SUPER
            # Not in any repository base: an external base, or `object` itself.
            return None, EXTERNAL if self._has_external_base(ctx.class_id) else BUILTIN

        target = self._evaluate(path, call.receiver, ctx, 0)
        if target is None:
            return None, UNRESOLVED

        callee = self._member(target, call.name, 0, for_call=True)
        if isinstance(callee, SymbolRef) and callee.kind in CALLABLE_KINDS:
            if call.receiver in ("self", "cls"):
                return callee.symbol_id, SELF
            if isinstance(target, ModuleRef):
                return callee.symbol_id, MODULE
            return callee.symbol_id, TYPED

        if isinstance(callee, Opaque):
            return None, callee.resolution
        return None, UNRESOLVED

    def base_ids(self, class_id: str) -> list[str | None]:
        """The resolved id of each base class, aligned with `Class.base_refs`."""
        if class_id in self._base_cache:
            return self._base_cache[class_id]

        # Mark as in progress first, so a cyclic hierarchy cannot recurse forever.
        self._base_cache[class_id] = []
        cls = self._classes[class_id]
        path = self._file_of[class_id]

        resolved: list[str | None] = []
        for ref in cls.base_refs:
            target = self._evaluate(path, ref, Context(None, NO_LOCALS), 0) if ref else None
            is_class = isinstance(target, SymbolRef) and target.kind == "class" and target.symbol_id != class_id
            resolved.append(target.symbol_id if is_class else None)  # type: ignore[union-attr]

        self._base_cache[class_id] = resolved
        return resolved

    def find_method(self, class_id: str, name: str, skip_own: bool = False) -> str | None:
        """Looks a method up along the class hierarchy, the way attribute lookup does."""
        order = self._mro(class_id)
        for owner in order[1:] if skip_own else order:
            method_id = self._methods.get(owner, {}).get(name)
            if method_id:
                return method_id
        return None

    # ------------------------------------------------------------- name lookup

    def _resolve_bare_call(self, path: str, ctx: Context, name: str) -> tuple[str | None, str]:
        if name == "cls" and ctx.class_id:
            return ctx.class_id, SELF  # `cls(...)` in a classmethod builds the class

        if name in ctx.local_types:
            # A parameter or local holding a callable: its target is unknowable.
            return None, UNRESOLVED

        target = self._resolve_name(path, name, 0)
        if isinstance(target, SymbolRef):
            if target.kind in ("function", "class"):
                how = DIRECT if self._file_of.get(target.symbol_id) == path else IMPORT
                return target.symbol_id, how
            target = self._deref(target, 0)

        if isinstance(target, Opaque):
            return None, target.resolution
        return None, UNRESOLVED

    def _resolve_name(self, path: str, name: str, depth: int) -> Target | None:
        """A name as the file's global namespace sees it."""
        if depth > MAX_DEPTH:
            return None

        definition = self._definitions.get(path, {}).get(name)
        if definition:
            return definition

        binding = self._bindings.get(path, {}).get(name)
        if binding:
            return self._resolve_binding(binding, depth + 1)

        for star in self._stars.get(path, ()):
            found = self._module_member(ModuleRef(star), name, depth + 1)
            if isinstance(found, (ModuleRef, SymbolRef)):
                return found

        if name in BUILTIN_NAMES:
            return BUILTIN_VALUE
        return None

    def _resolve_binding(self, binding: tuple[str, str, str], depth: int) -> Target | None:
        kind, module, name = binding

        if kind == "module":
            return ModuleRef(module) if self.modules.is_known(module) else EXTERNAL_VALUE

        submodule = f"{module}.{name}" if module else name
        if self.modules.is_known(submodule):
            return ModuleRef(submodule)  # `from pkg import module`
        if not self.modules.is_known(module):
            return EXTERNAL_VALUE
        return self._module_member(ModuleRef(module), name, depth + 1)

    def _module_member(self, module: ModuleRef, name: str, depth: int) -> Target | None:
        submodule = f"{module.name}.{name}"
        if self.modules.is_known(submodule):
            return ModuleRef(submodule)

        path = self.modules.file_for(module.name)
        if path is None:
            return None

        found = self._resolve_name(path, name, depth + 1)
        # A module attribute is never a builtin, even if the name is one.
        return None if found is BUILTIN_VALUE else found

    # -------------------------------------------------------------- evaluation

    def _evaluate(self, path: str, expr: str, ctx: Context, depth: int) -> Target | None:
        """Evaluates a receiver-style path such as `self.db`, `Service()` or `mod.obj.attr`."""
        if depth > MAX_DEPTH or not expr:
            return None

        head, *rest = expr.split(".")
        target = self._evaluate_head(path, head, ctx, depth)

        for segment in rest:
            if target is None or isinstance(target, Opaque):
                return target
            if segment.endswith("()"):
                member = self._member(target, segment[:-2], depth + 1, for_call=True)
                target = self._call_result(member, depth + 1)
            else:
                target = self._member(target, segment, depth + 1, for_call=False)

        return target

    def _evaluate_head(self, path: str, head: str, ctx: Context, depth: int) -> Target | None:
        called = head.endswith("()")
        name = head[:-2] if called else head

        if name in ("self", "cls") and not called:
            return SymbolRef(ctx.class_id, "class") if ctx.class_id else None

        if name in ctx.local_types:
            known = ctx.local_types[name]
            if known is None:
                return None
            # Local types were written with other locals already substituted,
            # so they are evaluated without the local namespace.
            target = self._evaluate(path, known, Context(ctx.class_id, NO_LOCALS), depth + 1)
        else:
            target = self._deref(self._resolve_name(path, name, depth + 1), depth + 1)

        return self._call_result(target, depth + 1) if called else target

    def _member(self, target: Target, name: str, depth: int, for_call: bool) -> Target | None:
        """Attribute `name` of `target`; with `for_call`, the thing being called."""
        if depth > MAX_DEPTH:
            return None

        if isinstance(target, Opaque):
            return target

        if isinstance(target, ModuleRef):
            found = self._module_member(target, name, depth + 1)
            return found if for_call else self._deref(found, depth + 1)

        if isinstance(target, SymbolRef) and target.kind == "class":
            class_id = target.symbol_id
            if for_call:
                method_id = self.find_method(class_id, name)
                if method_id:
                    return SymbolRef(method_id, "method")
                nested = self._find_nested(class_id, name)
                if nested:
                    return SymbolRef(nested, "class")
            else:
                found = self._instance_attribute(class_id, name, depth + 1)
                if found is not None:
                    return found
            return EXTERNAL_VALUE if self._has_external_base(class_id) else None

        return None

    def _instance_attribute(self, class_id: str, name: str, depth: int) -> Target | None:
        for owner in self._mro(class_id):
            cls = self._classes[owner]

            expr = cls.attr_types.get(name)
            if expr:
                found = self._evaluate(self._file_of[owner], expr, Context(class_id, NO_LOCALS), depth + 1)
                if found is not None:
                    return found

            method_id = self._methods.get(owner, {}).get(name)
            if method_id and self._is_property(method_id):
                return self._return_value(method_id, depth + 1)

            nested = self._nested.get(owner, {}).get(name)
            if nested:
                return SymbolRef(nested, "class")

        return None

    def _call_result(self, target: Target | None, depth: int) -> Target | None:
        """What calling `target` gives back: an instance of a class, or a function's return type."""
        if target is None or isinstance(target, Opaque):
            return target
        if isinstance(target, SymbolRef):
            if target.kind == "class":
                return target
            if target.kind in ("function", "method"):
                return self._return_value(target.symbol_id, depth + 1)
        return None

    def _return_value(self, function_id: str, depth: int) -> Target | None:
        func = self._functions.get(function_id)
        if func is None or not func.return_type or depth > MAX_DEPTH:
            return None
        ctx = Context(self._function_class[function_id], func.local_types)
        return self._evaluate(self._file_of[function_id], func.return_type, ctx, depth + 1)

    def _deref(self, target: Target | None, depth: int) -> Target | None:
        """Replaces a module-level variable with the value it holds, when that is known."""
        if not (isinstance(target, SymbolRef) and target.kind == "variable"):
            return target
        var = self._variables.get(target.symbol_id)
        if var is None or not var.value_type or depth > MAX_DEPTH:
            return None
        return self._evaluate(self._file_of[target.symbol_id], var.value_type, Context(None, NO_LOCALS), depth + 1)

    # -------------------------------------------------------- class hierarchy

    def _mro(self, class_id: str) -> list[str]:
        """The class followed by its repository bases, depth first, without repeats."""
        if class_id in self._mro_cache:
            return self._mro_cache[class_id]

        order: list[str] = []

        def visit(current: str, depth: int) -> None:
            if current in order or depth > MAX_DEPTH or current not in self._classes:
                return
            order.append(current)
            for base_id in self.base_ids(current):
                if base_id:
                    visit(base_id, depth + 1)

        visit(class_id, 0)
        self._mro_cache[class_id] = order
        return order

    def _has_external_base(self, class_id: str) -> bool:
        """True when some base class lies outside the repository (e.g. `ast.NodeVisitor`)."""
        for owner in self._mro(class_id):
            cls = self._classes[owner]
            if any(base_id is None for base_id in self.base_ids(owner)[: len(cls.base_refs)]):
                return True
        return False

    def _find_nested(self, class_id: str, name: str) -> str | None:
        for owner in self._mro(class_id):
            nested = self._nested.get(owner, {}).get(name)
            if nested:
                return nested
        return None

    def _is_property(self, method_id: str) -> bool:
        decorators = self._functions[method_id].decorators
        return any(d in PROPERTY_DECORATORS or d.endswith(".cached_property") for d in decorators)
