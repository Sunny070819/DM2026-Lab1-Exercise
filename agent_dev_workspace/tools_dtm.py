from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build Document-Term Matrix (DTM) tools bound to session."""

    @tool
    def build_dtm_tool(
        column_name: str = "text",
        ngram_range: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Constructs a Document-Term Matrix (DTM) using CountVectorizer.

        Stores the sparse feature matrix in session.feature_matrix, vocabulary terms in
        session.feature_names, and the fitted vectorizer in session.artifacts["count_vectorizer"].

        Args:
            column_name: Target text column in session.dataframe (default "text").
            ngram_range: Lower and upper boundary of n-grams as a list [min_n, max_n] (default [1, 1]).

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements,
            sparsity_pct, and vocabulary_preview.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in working DataFrame."}

        if ngram_range is None:
            effective_ngram = (1, 1)
        else:
            if (
                not isinstance(ngram_range, (list, tuple))
                or len(ngram_range) != 2
                or not all(isinstance(x, int) and x >= 1 for x in ngram_range)
                or ngram_range[0] > ngram_range[1]
            ):
                return {
                    "error": "ngram_range must be a list/tuple of 2 positive integers [min_n, max_n] with min_n <= max_n."
                }
            effective_ngram = (ngram_range[0], ngram_range[1])

        # Use temporary Series so session.dataframe is not mutated
        raw_text = session.dataframe[column_name].fillna("").astype(str)

        try:
            vectorizer = CountVectorizer(ngram_range=effective_ngram)
            feature_matrix = vectorizer.fit_transform(raw_text)
        except ValueError as e:
            return {"error": f"Failed to build DTM: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error building DTM: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements))
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
            "vocabulary_preview": feature_names[:10],
        }

        session.store_result(
            "build_dtm_tool",
            {"column_name": column_name, "ngram_range": list(effective_ngram)},
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates total term occurrences across all documents in the DTM.

        Sums columns of session.feature_matrix across all documents and returns terms
        sorted by frequency descending, with alphabetical tie-breaking.

        Args:
            top_n: Maximum number of top terms to return (default 20).

        Returns:
            Dict containing result_id, total_terms, and term_frequencies list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Call build_dtm_tool first."}

        if not isinstance(top_n, int) or top_n <= 0:
            return {"error": "top_n must be a positive integer."}

        # Calculate term totals
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_counts = [
            {"term": term, "count": int(cnt)}
            for term, cnt in zip(session.feature_names, counts)
        ]

        # Sort descending by count, ascending alphabetically by term for deterministic output
        sorted_terms = sorted(term_counts, key=lambda x: (-x["count"], x["term"]))

        effective_top_n = min(top_n, 100)
        result_list = sorted_terms[:effective_top_n]

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": result_list,
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Visualizes a positional slice of the raw Document-Term Matrix as a heatmap.

        Slices raw positional terms and documents from session.feature_matrix and sets
        a Seaborn heatmap on session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to slice (default 20).
            n_documents: Number of documents (rows) to slice (default 20).

        Returns:
            Dict containing result_id, n_documents_rendered, n_terms_rendered,
            terms list, documents list, and matrix slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Call build_dtm_tool first."}

        if not isinstance(n_terms, int) or n_terms <= 0:
            return {"error": "n_terms must be a positive integer."}

        if not isinstance(n_documents, int) or n_documents <= 0:
            return {"error": "n_documents must be a positive integer."}

        # Enforce maximum capping of 20x20 slice
        effective_docs = min(n_documents, 20, session.feature_matrix.shape[0])
        effective_terms = min(n_terms, 20, session.feature_matrix.shape[1])

        slice_matrix = session.feature_matrix[:effective_docs, :effective_terms].toarray()
        sliced_terms = session.feature_names[:effective_terms]
        doc_indices = list(range(effective_docs))

        # Create Seaborn heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            slice_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap (First {effective_docs} Docs x First {effective_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure without calling plt.show()
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents_rendered": effective_docs,
            "n_terms_rendered": effective_terms,
            "terms": sliced_terms,
            "documents": doc_indices,
            "matrix": slice_matrix.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [
        build_dtm_tool,
        term_frequency_tool,
        dtm_heatmap_tool,
    ]
