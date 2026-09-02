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
        classes, functions, _ = parse_code(code)  # <--- обновили распаковку
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
        code = '''
import os
import os.path as path
from typing import List, Optional as Opt
'''
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