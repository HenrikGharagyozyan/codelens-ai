"""Tests for VectorStore, with ChromaDB replaced by an in-memory double.

The real client would download an embedding model, so these tests verify the
translation layer we own: what we hand to Chroma and how we parse it back.
"""

import pytest

from codelens.indexer.chunker import Chunk
from codelens.indexer.vector_store import VectorStore


def make_chunk(name="run", start=1, end=4, path="src/app.py", content="def run(): ..."):
    return Chunk(
        chunk_id=f"{path}::{name}:{start}",
        file_path=path,
        symbol_name=name,
        start_line=start,
        end_line=end,
        content=content,
        symbol_type="function",
    )


@pytest.fixture
def store(fake_chroma_client, tmp_path):
    return VectorStore(tmp_path / "vectors")


class TestInitialisation:
    def test_creates_a_cosine_collection_at_the_given_path(self, fake_chroma_client, tmp_path):
        store = VectorStore(tmp_path / "vectors")

        assert fake_chroma_client[0].path == str(tmp_path / "vectors")
        assert store.collection.name == "code_chunks"
        assert store.collection.metadata == {"hnsw:space": "cosine"}

    def test_defaults_to_a_local_vector_directory(self, fake_chroma_client):
        VectorStore()

        assert fake_chroma_client[0].path == ".codelens_vector"


class TestAddChunks:
    def test_upserts_ids_documents_and_metadata_in_step(self, store):
        chunks = [make_chunk("run", 1, 4), make_chunk("stop", 6, 8)]

        store.add_chunks(chunks)
        call = store.collection.upserts[0]

        assert call["ids"] == [c.chunk_id for c in chunks]
        assert call["documents"] == [c.content for c in chunks]
        assert [m["symbol_name"] for m in call["metadatas"]] == ["run", "stop"]

    def test_metadata_carries_the_fields_the_retriever_reads(self, store):
        store.add_chunks([make_chunk("run", 1, 4)])
        meta = store.collection.upserts[0]["metadatas"][0]

        assert meta == {
            "chunk_id": "src/app.py::run:1",
            "file_path": "src/app.py",
            "symbol_name": "run",
            "symbol_type": "function",
            "start_line": 1,
            "end_line": 4,
        }

    def test_missing_line_numbers_become_the_sentinel(self, store):
        chunk = make_chunk()
        chunk.start_line = None
        chunk.end_line = None

        store.add_chunks([chunk])
        meta = store.collection.upserts[0]["metadatas"][0]

        assert meta["start_line"] == -1
        assert meta["end_line"] == -1

    def test_an_empty_list_does_not_touch_chroma(self, store):
        store.add_chunks([])

        assert store.collection.upserts == []


class TestClear:
    def test_drops_and_recreates_the_collection(self, store):
        store.add_chunks([make_chunk()])

        store.clear()

        assert store.client.deleted == ["code_chunks"]
        assert store.collection.upserts == []

    def test_survives_a_missing_collection(self, store):
        store.client.collections.clear()

        store.clear()

        assert store.collection.name == "code_chunks"


class TestSearch:
    def test_flattens_chroma_results_into_flat_dictionaries(self, store):
        store.collection.query_result = {
            "ids": [["a", "b"]],
            "documents": [["code a", "code b"]],
            "metadatas": [[{"symbol_name": "a"}, {"symbol_name": "b"}]],
            "distances": [[0.1, 0.4]],
        }

        results = store.search("how does a work")

        assert results == [
            {"id": "a", "document": "code a", "metadata": {"symbol_name": "a"}, "distance": 0.1},
            {"id": "b", "document": "code b", "metadata": {"symbol_name": "b"}, "distance": 0.4},
        ]

    def test_passes_the_limit_through_as_n_results(self, store):
        store.search("query", limit=3)

        assert store.collection.queries == [(["query"], 3)]

    def test_returns_an_empty_list_when_nothing_matches(self, store):
        assert store.search("nothing") == []
