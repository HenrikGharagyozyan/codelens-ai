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
        assert calls[0] == ("first_call", 3)
        assert calls[1] == ("second_call", 5)

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

        assert functions[0].calls == [("method", 2)]

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

        assert by_name["outer"].calls == [("outer_call", 3), ("after_inner", 8)]
        assert by_name["inner"].calls == [("inner_call", 6)]

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
