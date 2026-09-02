"""Tests for RepositoryScanner: which files are collected and how."""

from pathlib import Path

import pytest

from codelens.repository.models import File, Repository
from codelens.repository.scanner import RepositoryScanner


@pytest.fixture
def repo_dir(tmp_path):
    """A small repository tree with a .gitignore, a binary file and noise."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\nprint(os)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")

    # Excluded by the scanner's built-in defaults.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00\x01")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    # Excluded through .gitignore.
    (tmp_path / ".gitignore").write_text("*.log\nsecrets/\n", encoding="utf-8")
    (tmp_path / "debug.log").write_text("noise\n", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.txt").write_text("hunter2\n", encoding="utf-8")

    # Not decodable as UTF-8: must be skipped, not crash the scan.
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\xfd")

    return tmp_path


def scanned_paths(repo: Repository) -> set[str]:
    return {f.path.as_posix() for f in repo.files}


class TestScannerCollection:
    def test_returns_a_repository_rooted_at_the_absolute_path(self, repo_dir):
        repo = RepositoryScanner(repo_dir).scan()

        assert isinstance(repo, Repository)
        assert repo.root == repo_dir.resolve()
        assert repo.root.is_absolute()

    def test_accepts_a_string_path(self, repo_dir):
        repo = RepositoryScanner(str(repo_dir)).scan()

        assert repo.root == repo_dir.resolve()

    def test_collects_text_files_across_subdirectories(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert "src/app.py" in paths
        assert "README.md" in paths

    def test_paths_are_relative_to_the_root(self, repo_dir):
        repo = RepositoryScanner(repo_dir).scan()

        assert all(not f.path.is_absolute() for f in repo.files)

    def test_directories_are_not_reported_as_files(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert "src" not in paths
        assert all((repo_dir / p).is_file() for p in paths)

    def test_empty_directory_yields_no_files(self, tmp_path):
        repo = RepositoryScanner(tmp_path).scan()

        assert repo.files == []


class TestScannerExclusions:
    def test_skips_built_in_default_exclusions(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert not any(p.startswith("__pycache__/") for p in paths)
        assert not any(p.startswith(".venv/") for p in paths)

    def test_honours_gitignore_glob_patterns(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert "debug.log" not in paths

    def test_honours_gitignore_directory_patterns(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert "secrets/key.txt" not in paths

    def test_skips_binary_files_without_raising(self, repo_dir):
        paths = scanned_paths(RepositoryScanner(repo_dir).scan())

        assert "logo.png" not in paths

    def test_works_without_a_gitignore(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

        paths = scanned_paths(RepositoryScanner(tmp_path).scan())

        assert paths == {"app.py"}


class TestFileMetadata:
    def test_records_extension_size_and_line_count(self, repo_dir):
        repo = RepositoryScanner(repo_dir).scan()
        app = next(f for f in repo.files if f.path.as_posix() == "src/app.py")

        assert isinstance(app, File)
        assert app.language == "py"
        assert app.lines == 2
        assert app.size == (repo_dir / "src" / "app.py").stat().st_size

    def test_extensionless_files_are_language_unknown(self, repo_dir):
        repo = RepositoryScanner(repo_dir).scan()
        makefile = next(f for f in repo.files if f.path.as_posix() == "Makefile")

        assert makefile.language == "unknown"

    def test_counts_zero_lines_for_an_empty_file(self, tmp_path):
        (tmp_path / "empty.py").write_text("", encoding="utf-8")

        repo = RepositoryScanner(tmp_path).scan()

        assert repo.files[0].lines == 0
        assert repo.files[0].size == 0

    def test_final_line_without_newline_is_still_counted(self, tmp_path):
        (tmp_path / "a.py").write_text("one\ntwo", encoding="utf-8")

        repo = RepositoryScanner(tmp_path).scan()

        assert repo.files[0].lines == 2


class TestIgnoreSpec:
    def test_default_exclusions_are_loaded_even_without_gitignore(self, tmp_path):
        spec = RepositoryScanner(tmp_path).ignore_spec

        assert spec.match_file(".git/config")
        assert spec.match_file("node_modules/left-pad/index.js")
        assert spec.match_file("dist/bundle.js")
        assert not spec.match_file("src/app.py")

    def test_gitignore_rules_are_merged_with_the_defaults(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")

        spec = RepositoryScanner(tmp_path).ignore_spec

        assert spec.match_file("build/out.o")
        assert spec.match_file("scratch.tmp")


class TestRepositoryModels:
    def test_file_and_repository_carry_the_declared_fields(self):
        f = File(path=Path("src/app.py"), language="py", size=10, lines=2)
        repo = Repository(root=Path("/project"), files=[f])

        assert repo.files[0] is f
        assert repo.root == Path("/project")
