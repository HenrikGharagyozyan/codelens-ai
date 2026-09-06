import sys
from pathlib import Path

# Add the project root to PYTHONPATH so imports work
sys.path.append(str(Path(__file__).parent.parent / "src"))

from codelens.repository.db import DatabaseManager
from codelens.indexer.vector_store import VectorStore
from codelens.context.retriever import ContextRetriever

# Our reference dataset (Ground Truth)
# The query contains the question, and expected_file contains the file with the answer.
EVAL_DATASET = [
    {
        "query": "Where is the FTS5 virtual table created and triggered?",
        "expected_file": "src/codelens/repository/schema.py"
    },
    {
        "query": "How does the system combine vector and lexical search using RRF?",
        "expected_file": "src/codelens/context/retriever.py"
    },
    {
        "query": "Method that escapes quotes for exact keyword search in SQLite",
        "expected_file": "src/codelens/repository/db.py"
    },
    {
        "query": "How is the AST parser extracting classes and functions?",
        "expected_file": "src/codelens/parser/python_parser.py"  
    },
    {
        "query": "Prompts preamble used for context building",
        "expected_file": "src/codelens/llm/prompts.py"
    }
]

def calculate_recall_at_k(retriever: ContextRetriever, k_values: list[int] = [1, 2, 4]):
    print(f"Running Recall Evaluation on {len(EVAL_DATASET)} queries...\n")
    
    results_by_k = {k: 0 for k in k_values}
    
    for item in EVAL_DATASET:
        query = item["query"]
        expected_file = item["expected_file"]
        
        # Search with the maximum K so we can calculate all cutoffs afterward
        max_k = max(k_values)
        retrieved_chunks = retriever._hybrid_search(query, limit=max_k)
        
        # Extract file paths from the metadata of the retrieved chunks
        retrieved_files = [chunk["metadata"].get("file_path") for chunk in retrieved_chunks]
        
        print(f"Q: '{query}'")
        print(f"  Expected: {expected_file}")
        
        found_in_max_k = False
        # Corrected logic: check each K independently without break
        for k in k_values:
            top_k_files = retrieved_files[:k]
            if expected_file in top_k_files:
                results_by_k[k] += 1
                if k == max_k:
                    found_in_max_k = True

        if found_in_max_k:
             print(f"  ✅ Found in Top-{max_k}")
        else:
             print(f"  ❌ Not found in Top-{max_k}. Retrieved: {retrieved_files}")
        print("-" * 40)

    # Print the final metrics
    print("\n=== EVALUATION RESULTS ===")
    for k in k_values:
        recall = (results_by_k[k] / len(EVAL_DATASET)) * 100
        print(f"Recall@{k}: {recall:.1f}% ({results_by_k[k]}/{len(EVAL_DATASET)} queries)")

if __name__ == "__main__":
    db = DatabaseManager()
    vector_store = VectorStore()
    retriever = ContextRetriever(db, vector_store)
    
    try:
        calculate_recall_at_k(retriever, k_values=[1, 2, 4])
    finally:
        db.close()