"""Regression tests for Python AST Parser.

Locks in fixes for:
- Nested classes and class context isolation
- Base classes extracted from ast.Attribute (e.g., typing.Generic)
- Function calls tracking accurate line numbers (callee_name, call_line)
- Imports and Docstrings extraction
"""

import pytest

from codelens.parser.python_parser import parse_python_file


@pytest.fixture
def parse_code(tmp_path):
    """Fixture to write a temporary Python code snippet and parse it."""

    def _parse(code: str):
        file_path = tmp_path / "sample.py"
        file_path.write_text(code, encoding="utf-8")
        return parse_python_file(file_path)

    return _parse


class TestPythonParser:
    def test_nested_classes_do_not_corrupt_outer_context(self, parse_code):
        code = """
class Outer:
    def outer_method(self):
        pass

    class Inner:
        def inner_method(self):
            pass
"""
        classes, functions, _ = parse_code(code)
        class_map = {c.name: c for c in classes}

        assert "Outer" in class_map
        assert "Inner" in class_map

        outer_methods = [m.name for m in class_map["Outer"].methods]
        inner_methods = [m.name for m in class_map["Inner"].methods]

        assert outer_methods == ["outer_method"]
        assert inner_methods == ["inner_method"]
        assert functions == []

    def test_extracts_attribute_and_name_base_classes(self, parse_code):
        code = """
import typing
import base_module

class MyView(typing.Generic, base_module.BaseClass, LocalBase):
    def get(self):
        pass
"""
        classes, _, _ = parse_code(code)
        assert len(classes) == 1
        cls = classes[0]

        assert "Generic" in cls.bases
        assert "BaseClass" in cls.bases
        assert "LocalBase" in cls.bases

    def test_records_call_line_numbers_accurately(self, parse_code):
        code = """
def process_data():
    first_call()     # line 3

    second_call()    # line 5
"""
        _, functions, _ = parse_code(code)
        assert len(functions) == 1
        fn = functions[0]

        calls = fn.calls
        assert len(calls) == 2
        assert calls[0] == ("first_call", 3, None)
        assert calls[1] == ("second_call", 5, None)

    def test_async_functions_and_methods(self, parse_code):
        code = """
async def async_top_level():
    await helper()

class Worker:
    async def fetch(self):
        pass
"""
        classes, functions, _ = parse_code(code)

        assert len(functions) == 1
        assert functions[0].name == "async_top_level"

        assert len(classes) == 1
        assert classes[0].methods[0].name == "fetch"

    def test_extracts_docstrings_from_classes(self, parse_code):
        """Docstrings should be cleanly extracted for LLM context."""
        code = '''
class Worker:
    """This is a worker class.
    It does things."""
    def work(self): pass
'''
        classes, _, _ = parse_code(code)
        assert len(classes) == 1
        assert classes[0].docstring.startswith("This is a worker class.")

    def test_extracts_imports(self, parse_code):
        """Imports (both regular and from) must be captured."""
        code = """
import os
import os.path as path
from typing import List, Optional as Opt
"""
        _, _, imports = parse_code(code)

        assert len(imports) == 4

        # import os
        assert imports[0].name == "os"
        assert imports[0].module is None

        # import os.path as path
        assert imports[1].name == "os.path"
        assert imports[1].alias == "path"

        # from typing import List
        assert imports[2].module == "typing"
        assert imports[2].name == "List"

        # from typing import Optional as Opt
        assert imports[3].module == "typing"
        assert imports[3].name == "Optional"
        assert imports[3].alias == "Opt"

    def test_docstrings_are_none_when_absent(self, parse_code):
        classes, functions, _ = parse_code("class A:\n    pass\n\ndef f():\n    pass\n")

        assert classes[0].docstring is None
        assert functions[0].docstring is None

    def test_extracts_function_docstrings(self, parse_code):
        _, functions, _ = parse_code('def f():\n    """Does a thing."""\n')

        assert functions[0].docstring == "Does a thing."

    def test_records_argument_names_in_order(self, parse_code):
        _, functions, _ = parse_code("def f(a, b, c=1):\n    pass\n")

        assert functions[0].args == ["a", "b", "c"]

    def test_methods_keep_the_self_argument(self, parse_code):
        classes, _, _ = parse_code("class A:\n    def m(self, x):\n        pass\n")

        assert classes[0].methods[0].args == ["self", "x"]

    def test_records_line_ranges_for_chunking(self, parse_code):
        _, functions, _ = parse_code("def f():\n    a = 1\n    return a\n")

        assert functions[0].line_number == 1
        assert functions[0].end_line_number == 3

    def test_flags_async_functions(self, parse_code):
        _, functions, _ = parse_code("async def f():\n    pass\n\ndef g():\n    pass\n")

        assert functions[0].is_async is True
        assert functions[1].is_async is False

    def test_a_class_without_bases_has_an_empty_list(self, parse_code):
        classes, _, _ = parse_code("class A:\n    pass\n")

        assert classes[0].bases == []

    def test_an_unsupported_base_expression_is_labelled(self, parse_code):
        classes, _, _ = parse_code("class A(make_base()):\n    pass\n")

        assert classes[0].bases == ["UnknownBase"]

    def test_records_method_calls_by_attribute_name(self, parse_code):
        _, functions, _ = parse_code("def f():\n    obj.method()\n")

        assert functions[0].calls == [("method", 2, "obj")]

    def test_calls_are_attributed_to_the_innermost_function(self, parse_code):
        code = """
def outer():
    outer_call()

    def inner():
        inner_call()

    after_inner()
"""
        _, functions, _ = parse_code(code)
        by_name = {f.name: f for f in functions}

        # The nested 'inner' function should not be exposed as a top-level symbol.
        assert "inner" not in by_name
        # Its calls should be attributed to the parent function in AST traversal order.
        assert by_name["outer"].calls == [("outer_call", 3, None), ("inner_call", 6, None), ("after_inner", 8, None)]

    def test_module_level_calls_are_not_attributed_to_any_function(self, parse_code):
        _, functions, _ = parse_code("print('hi')\n\ndef f():\n    pass\n")

        assert functions[0].calls == []

    def test_decorated_functions_are_still_captured(self, parse_code):
        code = """
import functools

@functools.cache
def cached():
    pass
"""
        _, functions, _ = parse_code(code)

        assert functions[0].name == "cached"

    def test_captures_relative_imports(self, parse_code):
        _, _, imports = parse_code("from . import models\nfrom .db import connect\n")

        assert imports[0].module == ""
        assert imports[0].name == "models"
        assert imports[1].module == "db"
        assert imports[1].name == "connect"

    def test_file_path_is_recorded_on_every_symbol(self, parse_code, tmp_path):
        classes, functions, imports = parse_code(
            "import os\n\nclass A:\n    def m(self):\n        pass\n\ndef f():\n    pass\n"
        )
        expected = str(tmp_path / "sample.py")

        assert classes[0].file_path == expected
        assert classes[0].methods[0].file_path == expected
        assert functions[0].file_path == expected
        assert imports[0].file_path == expected


class TestParserResilience:
    def test_a_syntax_error_yields_empty_results_instead_of_raising(self, parse_code):
        assert parse_code("def broken(:\n") == ([], [], [])

    def test_an_unreadable_file_yields_empty_results(self, tmp_path):
        missing = tmp_path / "gone.py"

        assert parse_python_file(missing) == ([], [], [])

    def test_a_binary_file_yields_empty_results(self, tmp_path):
        binary = tmp_path / "blob.py"
        binary.write_bytes(b"\xff\xfe\x00\x01")

        assert parse_python_file(binary) == ([], [], [])

    def test_an_empty_file_yields_empty_results(self, parse_code):
        assert parse_code("") == ([], [], [])

    def test_a_comment_only_file_yields_empty_results(self, parse_code):
        assert parse_code("# nothing to see here\n") == ([], [], [])

    def test_record_as_overrides_the_path_stored_on_symbols(self, tmp_path):
        """The indexer uses this so nothing has to rewrite `file_path` later."""
        source = tmp_path / "sample.py"
        source.write_text("class A:\n    def m(self):\n        pass\n\ndef f():\n    pass\n")

        classes, functions, _ = parse_python_file(source, record_as="src/sample.py")

        assert classes[0].file_path == "src/sample.py"
        assert classes[0].methods[0].file_path == "src/sample.py"
        assert functions[0].file_path == "src/sample.py"

    def test_record_as_also_applies_to_imports(self, tmp_path):
        source = tmp_path / "sample.py"
        source.write_text("import os\n")

        _, _, imports = parse_python_file(source, record_as="src/sample.py")

        assert imports[0].file_path == "src/sample.py"


class TestGraphMetadata:
    """What the parser records for the call graph: receivers, names, signatures and types."""

    @pytest.fixture
    def parse_full(self, tmp_path):
        from codelens.parser.python_parser import parse_file

        def _parse(code: str):
            path = tmp_path / "sample.py"
            path.write_text(code, encoding="utf-8")
            return parse_file(path)

        return _parse

    def test_calls_record_what_they_were_made_on(self, parse_full):
        code = """
class A(Base):
    def m(self, db: Database):
        self.save()
        self.db.query()
        Service().run()
        super().m()
        items[0].pop()
        db.close()
        ", ".join([])
"""
        method = parse_full(code).classes[0].methods[0]
        calls = {(c.name, c.receiver) for c in method.calls}

        assert {
            ("save", "self"),
            ("query", "self.db"),
            ("Service", None),
            ("run", "Service()"),
            ("m", "super()"),
            ("pop", "<expr>"),
            ("close", "db"),
            ("join", "str()"),
        } <= calls

    def test_nested_classes_get_qualified_names(self, parse_full):
        code = "class Outer:\n    class Inner:\n        def go(self):\n            pass\n"
        classes = {c.name: c for c in parse_full(code).classes}

        assert classes["Inner"].qualname == "Outer.Inner"
        assert classes["Inner"].methods[0].qualname == "Outer.Inner.go"

    def test_signature_decorators_and_return_annotation(self, parse_full):
        code = "class A:\n    @property\n    def v(self) -> int:\n        return 1\n"
        method = parse_full(code).classes[0].methods[0]

        assert method.signature == "def v(self) -> int"
        assert method.decorators == ["property"]
        assert method.returns == "int"

    def test_class_signature_keeps_bases_and_keywords(self, parse_full):
        cls = parse_full("class A(Base, metaclass=Meta):\n    pass\n").classes[0]

        assert cls.signature == "class A(Base, metaclass=Meta)"
        assert cls.base_refs == ["Base"]

    def test_every_kind_of_parameter_is_listed(self, parse_full):
        func = parse_full("def f(a, /, b, *args, c, **kwargs):\n    pass\n").functions[0]

        assert func.args == ["a", "b", "*args", "c", "**kwargs"]

    def test_parameter_annotations_become_local_types(self, parse_full):
        code = "def f(a: Db, b: Optional[Db], c: 'Db', d: Db | None, e, f: list[Db]):\n    pass\n"
        local_types = parse_full(code).functions[0].local_types

        assert local_types == {"a": "Db", "b": "Db", "c": "Db", "d": "Db", "e": None, "f": "list"}

    def test_self_is_not_a_local(self, parse_full):
        method = parse_full("class A:\n    def m(self, x: Db):\n        pass\n").classes[0].methods[0]

        assert "self" not in method.local_types

    def test_locals_built_by_constructors_and_loop_variables(self, parse_full):
        code = (
            "def f(rows):\n"
            "    db = Database()\n"
            "    for row in rows:\n"
            "        pass\n"
            "    with open(p) as fh:\n"
            "        pass\n"
        )
        local_types = parse_full(code).functions[0].local_types

        assert local_types["db"] == "Database()"
        assert local_types["row"] is None
        assert local_types["fh"] is None

    def test_self_attribute_types_follow_the_parameter_annotation(self, parse_full):
        code = """
class Service:
    root: Path

    def __init__(self, db: Database | None = None):
        self.db = db if db is not None else Database()
        self.cache = Cache()
"""
        attr_types = parse_full(code).classes[0].attr_types

        assert attr_types == {"root": "Path", "db": "Database", "cache": "Cache()"}

    def test_return_type_is_inferred_when_not_annotated(self, parse_full):
        parsed = parse_full("def make():\n    return Service()\n\ndef typed() -> Db:\n    return Service()\n")
        by_name = {f.name: f for f in parsed.functions}

        assert by_name["make"].return_type == "Service()"
        assert by_name["typed"].return_type == "Db"

    def test_module_level_names_are_variables_not_functions(self, parse_full):
        parsed = parse_full("DB_PATH = '.db'\nconsole = Console()\n\ndef f():\n    pass\n")

        assert [f.name for f in parsed.functions] == ["f"]
        assert {v.name: v.value_type for v in parsed.variables} == {"DB_PATH": "str()", "console": "Console()"}
        assert parsed.module.top_level_names == ["DB_PATH", "console", "f"]

    def test_relative_import_level_is_recorded(self, parse_full):
        imports = parse_full("from ..pkg import x\nfrom . import y\nimport z\n").imports

        assert [(i.module, i.name, i.level) for i in imports] == [("pkg", "x", 2), ("", "y", 1), (None, "z", 0)]

    def test_decorator_calls_are_not_calls_made_by_the_function(self, parse_full):
        func = parse_full("@app.command()\ndef f():\n    pass\n").functions[0]

        assert func.calls == []

    def test_a_class_inside_a_function_is_local(self, parse_full):
        code = "def factory():\n    class Local:\n        def run(self):\n            helper()\n    return Local\n"
        parsed = parse_full(code)

        assert parsed.classes == []
        assert [c.name for c in parsed.functions[0].calls] == ["helper"]
