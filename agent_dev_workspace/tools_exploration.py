import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    """Factory function to instantiate tools with access to the shared session."""

    @tool
    def cosine_similarity_tool(
        doc_idx1: int,
        doc_idx2: int,
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents using raw feature counts.

        Args:
            doc_idx1: Index of the first document.
            doc_idx2: Index of the second document.

        Returns:
            Dict containing document indices and computed cosine similarity score.
        """
        # Validate DTM existence
        if session.feature_matrix is None:
            return {"error": "No feature_matrix (DTM) found in session. Please run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]

        # Validate document indices
        if doc_idx1 < 0 or doc_idx1 >= n_docs:
            return {"error": f"doc_idx1 out of range: {doc_idx1}. Valid range is 0 to {n_docs - 1}."}

        if doc_idx2 < 0 or doc_idx2 >= n_docs:
            return {"error": f"doc_idx2 out of range: {doc_idx2}. Valid range is 0 to {n_docs - 1}."}

        # Extract document rows (scipy sparse or numpy array)
        doc1_vec = session.feature_matrix[doc_idx1]
        doc2_vec = session.feature_matrix[doc_idx2]

        # Compute cosine similarity
        sim_matrix = cosine_similarity(doc1_vec, doc2_vec)
        similarity = float(sim_matrix[0, 0])

        # Handle empty/zero vectors cleanly to ensure exact 0.0 float instead of NaN
        if np.isnan(similarity):
            similarity = 0.0

        args_recorded = {
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
        }

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": round(similarity, 4),
        }

        session.store_result("cosine_similarity_tool", args_recorded, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation matrix over the top-N terms by variance,

        and plots a correlation heatmap on session.pending_figure.

        Args:
            top_n: Number of top terms by variance to compute correlation for (default 20).

        Returns:
            Dict containing top terms, correlation matrix, and stored result id.
        """
        # Validate DTM and feature_names existence
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names found in session. Please run build_dtm_tool first."}

        if top_n <= 0:
            return {"error": f"top_n must be positive, got {top_n}."}

        # Convert sparse matrix to dense array for variance computation
        dtm_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.array(session.feature_matrix)
        feature_names = list(session.feature_names)

        n_features = dtm_dense.shape[1]
        if n_features == 0:
            return {"error": "Document-term matrix has no features."}

        # 1. Compute term variances across all documents
        variances = np.var(dtm_dense, axis=0)

        # Take top_n by variance descending
        actual_k = min(top_n, n_features)
        top_indices = np.argsort(variances)[::-1][:actual_k]

        top_terms = [feature_names[i] for i in top_indices]
        top_dtm_slice = dtm_dense[:, top_indices]

        # 2. Compute Pearson correlation matrix across top terms (columns)
        # np.corrcoef expects variables as rows, so transpose top_dtm_slice
        corr_matrix = np.corrcoef(top_dtm_slice, rowvar=False)

        # Handle edge case where corrcoef returns scalar or 1D array if actual_k == 1
        if actual_k == 1:
            corr_matrix = np.array([[1.0]])

        # Fill NaNs if constant features produce 0 variance
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        # 3. Build heatmap of correlation matrix and set pending_figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Format full_report DataFrame
        corr_df = pd.DataFrame(corr_matrix_rounded, index=top_terms, columns=top_terms)

        args_recorded = {"top_n": top_n}

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_matrix_rounded,
        }

        session.artifacts["feature_correlation_matrix"] = summary
        session.store_result("feature_correlation_matrix_tool", args_recorded, summary, full_report=corr_df)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
