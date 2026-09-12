"""Tests for the call resolver: every rule it follows, and every case where it must refuse to guess.

Each test builds a tiny repository on disk, parses it for real and asks the
resolver about one call site. A wrong edge is worse than a missing one, so the
"must not resolve" cases matter as much as the positive ones.
"""

import textwrap

import pytest

from codelens.graph.resolver import (
    BUILTIN,
    DIRECT,
    EXTERNAL,
    IMPORT,
    MODULE,
    SELF,
    SUPER,
    TYPED,
    UNRESOLVED,
    ModuleRegistry,
    SymbolTable,
    module_name,
)
from codelens.indexer.runner import CodebaseIndexer
from codelens.parser.python_parser import parse_file
from tests.conftest import FakeVectorStore


class Repo:
    """A parsed repository plus helpers to look up call sites by name."""

    def __init__(self, table: SymbolTable, functions: dict):
        self.table = table
        self.functions = functions

    def resolve(self, caller_id: str, callee_name: str, receiver: str | None = ...):
        func = self.functions[caller_id]
        calls = [c for c in func.calls if c.name == callee_name and (receiver is ... or c.receiver == receiver)]
        assert calls, f"{caller_id} makes no call to {callee_name!r}: {func.calls}"
        return self.table.resolve_call(caller_id, calls[0])


@pytest.fixture
def build(tmp_path, db):
    def _build(files: dict[str, str]) -> Repo:
        parsed = {}
        for rel_path, code in files.items():
            path = tmp_path / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(code), encoding="utf-8")
            parsed[rel_path] = parse_file(path, record_as=rel_path)

        CodebaseIndexer(str(tmp_path), db=db, vector_store=FakeVectorStore())._assign_ids(parsed)

        functions = {}
        for p in parsed.values():
            for func in p.functions:
                functions[func.symbol_id] = func
            for cls in p.classes:
                for method in cls.methods:
                    functions[method.symbol_id] = method

        return Repo(SymbolTable(parsed), functions)

    return _build


class TestModuleRegistry:
    def test_module_names_follow_the_path(self):
        assert module_name("src/codelens/db.py") == "src.codelens.db"
        assert module_name("pkg/__init__.py") == "pkg"

    def test_a_file_is_reachable_by_every_suffix_of_its_dotted_path(self):
        registry = ModuleRegistry(["src/codelens/db.py"])

        assert registry.file_for("codelens.db") == "src/codelens/db.py"
        assert registry.file_for("db") == "src/codelens/db.py"
        assert registry.is_known("codelens")

    def test_an_ambiguous_suffix_resolves_to_nothing(self):
        registry = ModuleRegistry(["a/models.py", "b/models.py"])

        assert registry.file_for("models") is None
        assert registry.file_for("a.models") == "a/models.py"


class TestBareCalls:
    def test_a_function_defined_in_the_same_file(self, build):
        repo = build({"a.py": "def helper():\n    pass\n\ndef main():\n    helper()\n"})

        assert repo.resolve("a.py::main", "helper") == ("a.py::helper", DIRECT)

    def test_a_function_imported_from_another_module(self, build):
        repo = build(
            {
                "pkg/db.py": "def connect():\n    pass\n",
                "app.py": "from pkg.db import connect\n\ndef main():\n    connect()\n",
            }
        )

        assert repo.resolve("app.py::main", "connect") == ("pkg/db.py::connect", IMPORT)

    def test_an_import_rooted_below_the_repository_root(self, build):
        """`src/` layouts import `codelens.db`, not `src.codelens.db`."""
        repo = build(
            {
                "src/codelens/db.py": "def connect():\n    pass\n",
                "src/codelens/app.py": "from codelens.db import connect\n\ndef main():\n    connect()\n",
            }
        )

        assert repo.resolve("src/codelens/app.py::main", "connect") == ("src/codelens/db.py::connect", IMPORT)

    def test_a_relative_import(self, build):
        repo = build(
            {
                "pkg/__init__.py": "",
                "pkg/models.py": "def make():\n    pass\n",
                "pkg/service.py": "from .models import make\n\ndef run():\n    make()\n",
            }
        )

        assert repo.resolve("pkg/service.py::run", "make") == ("pkg/models.py::make", IMPORT)

    def test_a_re_export_through_a_package_init(self, build):
        repo = build(
            {
                "pkg/__init__.py": "from .impl import connect\n",
                "pkg/impl.py": "def connect():\n    pass\n",
                "app.py": "from pkg import connect\n\ndef main():\n    connect()\n",
            }
        )

        assert repo.resolve("app.py::main", "connect") == ("pkg/impl.py::connect", IMPORT)

    def test_calling_a_class_links_to_the_class(self, build):
        repo = build({"a.py": "class Service:\n    pass\n\ndef main():\n    Service()\n"})

        assert repo.resolve("a.py::main", "Service") == ("a.py::Service", DIRECT)

    def test_builtins_are_labelled_not_linked(self, build):
        repo = build({"a.py": "def main(items):\n    len(items)\n"})

        assert repo.resolve("a.py::main", "len") == (None, BUILTIN)

    def test_third_party_imports_are_labelled_external(self, build):
        repo = build({"a.py": "from rich.console import Console\n\ndef main():\n    Console()\n"})

        assert repo.resolve("a.py::main", "Console") == (None, EXTERNAL)

    def test_a_parameter_holding_a_callable_is_not_guessed(self, build):
        repo = build({"a.py": "def helper():\n    pass\n\ndef apply(helper):\n    helper()\n"})

        assert repo.resolve("a.py::apply", "helper") == (None, UNRESOLVED)

    def test_a_nested_function_shadows_a_global_of_the_same_name(self, build):
        repo = build(
            {
                "a.py": """
                def helper():
                    pass

                def main():
                    def helper():
                        pass
                    helper()
                """
            }
        )

        assert repo.resolve("a.py::main", "helper") == (None, UNRESOLVED)


class TestModuleCalls:
    def test_a_function_through_an_aliased_module_import(self, build):
        repo = build(
            {
                "pkg/db.py": "def connect():\n    pass\n",
                "app.py": "import pkg.db as db\n\ndef main():\n    db.connect()\n",
            }
        )

        assert repo.resolve("app.py::main", "connect") == ("pkg/db.py::connect", MODULE)

    def test_a_function_through_a_dotted_import(self, build):
        repo = build(
            {
                "pkg/db.py": "def connect():\n    pass\n",
                "app.py": "import pkg.db\n\ndef main():\n    pkg.db.connect()\n",
            }
        )

        assert repo.resolve("app.py::main", "connect") == ("pkg/db.py::connect", MODULE)

    def test_from_package_import_module(self, build):
        repo = build(
            {
                "pkg/__init__.py": "",
                "pkg/models.py": "def make():\n    pass\n",
                "pkg/service.py": "from . import models\n\ndef run():\n    models.make()\n",
            }
        )

        assert repo.resolve("pkg/service.py::run", "make") == ("pkg/models.py::make", MODULE)

    def test_a_standard_library_module_is_external(self, build):
        repo = build({"a.py": "import os\n\ndef main():\n    os.path.join('a', 'b')\n"})

        assert repo.resolve("a.py::main", "join") == (None, EXTERNAL)

    def test_an_ambiguous_module_is_not_guessed(self, build):
        repo = build(
            {
                "a/models.py": "def make():\n    pass\n",
                "b/models.py": "def make():\n    pass\n",
                "app.py": "import models\n\ndef main():\n    models.make()\n",
            }
        )

        assert repo.resolve("app.py::main", "make") == (None, UNRESOLVED)

    def test_a_loop_variable_is_not_the_module_of_the_same_name(self, build):
        repo = build(
            {
                "db.py": "def query():\n    pass\n",
                "app.py": "import db\n\ndef main(shards):\n    for db in shards:\n        db.query()\n",
            }
        )

        assert repo.resolve("app.py::main", "query") == (None, UNRESOLVED)


class TestMethodCalls:
    def test_a_method_on_self(self, build):
        repo = build(
            {
                "a.py": """
                class Service:
                    def run(self):
                        self.helper()

                    def helper(self):
                        pass
                """
            }
        )

        assert repo.resolve("a.py::Service.run", "helper") == ("a.py::Service.helper", SELF)

    def test_an_inherited_method_from_a_base_in_another_file(self, build):
        repo = build(
            {
                "base.py": "class Base:\n    def save(self):\n        pass\n",
                "child.py": "from base import Base\n\nclass Child(Base):\n    def run(self):\n        self.save()\n",
            }
        )

        assert repo.resolve("child.py::Child.run", "save") == ("base.py::Base.save", SELF)
        assert repo.table.base_ids("child.py::Child") == ["base.py::Base"]

    def test_super_skips_the_class_itself(self, build):
        repo = build(
            {
                "a.py": """
                class Base:
                    def save(self):
                        pass

                class Child(Base):
                    def save(self):
                        super().save()
                """
            }
        )

        assert repo.resolve("a.py::Child.save", "save") == ("a.py::Base.save", SUPER)

    def test_super_into_an_external_base_is_external(self, build):
        repo = build(
            {
                "a.py": """
                import ast

                class Visitor(ast.NodeVisitor):
                    def visit(self, node):
                        super().visit(node)
                """
            }
        )

        assert repo.resolve("a.py::Visitor.visit", "visit") == (None, EXTERNAL)

    def test_a_missing_method_of_a_class_with_an_external_base_is_external(self, build):
        repo = build(
            {
                "a.py": """
                import ast

                class Visitor(ast.NodeVisitor):
                    def run(self, node):
                        self.generic_visit(node)
                """
            }
        )

        assert repo.resolve("a.py::Visitor.run", "generic_visit") == (None, EXTERNAL)

    def test_cls_in_a_classmethod_builds_the_class(self, build):
        repo = build(
            {
                "a.py": """
                class Node:
                    @classmethod
                    def empty(cls):
                        return cls()
                """
            }
        )

        assert repo.resolve("a.py::Node.empty", "cls") == ("a.py::Node", SELF)

    def test_a_cyclic_hierarchy_does_not_hang(self, build):
        repo = build(
            {
                "a.py": """
                class A(B):
                    def run(self):
                        self.missing()

                class B(A):
                    pass
                """
            }
        )

        assert repo.resolve("a.py::A.run", "missing") == (None, UNRESOLVED)


class TestTypedCalls:
    DATABASE = """
    class Database:
        def query(self):
            pass
    """

    def test_an_annotated_parameter(self, build):
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": "from db import Database\n\ndef run(db: Database):\n    db.query()\n",
            }
        )

        assert repo.resolve("app.py::run", "query") == ("db.py::Database.query", TYPED)

    @pytest.mark.parametrize("annotation", ["Optional[Database]", "Database | None", '"Database"'])
    def test_optional_and_string_annotations(self, build, annotation):
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": (
                    "from typing import Optional\nfrom db import Database\n\n"
                    f"def run(db: {annotation}):\n    db.query()\n"
                ),
            }
        )

        assert repo.resolve("app.py::run", "query") == ("db.py::Database.query", TYPED)

    def test_a_local_built_by_a_constructor(self, build):
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": "from db import Database\n\ndef run():\n    db = Database()\n    db.query()\n",
            }
        )

        assert repo.resolve("app.py::run", "query") == ("db.py::Database.query", TYPED)

    def test_a_self_attribute_assigned_in_init_with_a_fallback(self, build):
        """The CodebaseIndexer pattern: `self.db = db if db is not None else Database()`."""
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": """
                from db import Database

                class Service:
                    def __init__(self, db: Database | None = None):
                        self.db = db if db is not None else Database()

                    def run(self):
                        self.db.query()
                """,
            }
        )

        assert repo.resolve("app.py::Service.run", "query") == ("db.py::Database.query", TYPED)

    def test_a_lazy_property_behind_a_parameter(self, build):
        """The AppContext pattern: `ctx.db.query()` through a caching property."""
        repo = build(
            {
                "db.py": self.DATABASE,
                "context.py": """
                class AppContext:
                    _db = None

                    @property
                    def db(self):
                        if self._db is None:
                            from db import Database
                            self._db = Database()
                        return self._db
                """,
                "cli.py": "from context import AppContext\n\ndef command(ctx: AppContext):\n    ctx.db.query()\n",
            }
        )

        assert repo.resolve("cli.py::command", "query") == ("db.py::Database.query", TYPED)

    def test_a_class_level_annotation(self, build):
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": """
                from dataclasses import dataclass
                from db import Database

                @dataclass
                class Service:
                    db: Database

                    def run(self):
                        self.db.query()
                """,
            }
        )

        assert repo.resolve("app.py::Service.run", "query") == ("db.py::Database.query", TYPED)

    def test_a_function_return_annotation(self, build):
        repo = build(
            {
                "db.py": self.DATABASE,
                "app.py": (
                    "from db import Database\n\n"
                    "def open_db() -> Database:\n    ...\n\n"
                    "def run():\n    open_db().query()\n"
                ),
            }
        )

        assert repo.resolve("app.py::run", "query") == ("db.py::Database.query", TYPED)

    def test_a_module_level_singleton(self, build):
        repo = build(
            {
                "db.py": textwrap.dedent(self.DATABASE) + "\ndatabase = Database()\n",
                "app.py": "from db import database\n\ndef run():\n    database.query()\n",
            }
        )

        assert repo.resolve("app.py::run", "query") == ("db.py::Database.query", TYPED)

    def test_the_right_method_among_same_named_ones(self, build):
        """The failure mode of name matching: two `connect` methods, one call."""
        repo = build(
            {
                "a.py": """
                class Cache:
                    def connect(self):
                        pass

                class Database:
                    def connect(self):
                        pass

                def run(db: Database):
                    db.connect()
                """
            }
        )

        assert repo.resolve("a.py::run", "connect") == ("a.py::Database.connect", TYPED)

    def test_methods_of_builtin_containers_are_builtin(self, build):
        repo = build({"a.py": "def run(names: list[str]):\n    names.append('x')\n    ', '.join(names)\n"})

        assert repo.resolve("a.py::run", "append") == (None, BUILTIN)
        assert repo.resolve("a.py::run", "join") == (None, BUILTIN)

    def test_an_untyped_receiver_stays_unresolved(self, build):
        repo = build({"a.py": "class Database:\n    def query(self):\n        pass\n\ndef run(db):\n    db.query()\n"})

        assert repo.resolve("a.py::run", "query") == (None, UNRESOLVED)

    def test_an_unfollowable_receiver_stays_unresolved(self, build):
        repo = build({"a.py": "def run(items):\n    items[0].save()\n"})

        assert repo.resolve("a.py::run", "save") == (None, UNRESOLVED)
