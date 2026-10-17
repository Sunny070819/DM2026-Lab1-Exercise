from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build feature filtering tools bound to session."""

    @tool
    def variance_filter_tool(threshold: float = 0.1) -> Dict[str, Any]:
        """Filters terms from the Document-Term Matrix whose variance is below a threshold.

        Updates session.feature_matrix and session.feature_names in-place.

        Args:
            threshold: Minimum population variance required to keep a term (default 0.1).

        Returns:
            Dict containing result_id, terms_kept, terms_removed, kept_terms preview,
            and removed_terms preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Call build_dtm_tool first."}

        if not isinstance(threshold, (int, float)) or threshold < 0:
            return {"error": "threshold must be a non-negative float."}

        # Sparse-friendly population variance: Var(X) = E[X^2] - (E[X])^2
        X = session.feature_matrix
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        # Retain terms with variance >= threshold
        keep_mask = variances >= threshold
        kept_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        if len(kept_indices) == 0:
            return {
                "error": f"Filter threshold {threshold} removes all terms. DTM was not modified."
            }

        original_feature_names = list(session.feature_names)
        kept_terms = [original_feature_names[i] for i in kept_indices]
        removed_terms = [original_feature_names[i] for i in removed_indices]

        # Store report DataFrame using full original term list BEFORE modifying session
        report_df = pd.DataFrame(
            {"term": original_feature_names, "variance": variances}
        ).sort_values(by="variance", ascending=False)

        # Update session state in-place
        session.feature_matrix = session.feature_matrix[:, kept_indices]
        session.feature_names = kept_terms

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "terms_kept": len(kept_terms),
            "terms_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
            "threshold": float(threshold),
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Filters DTM terms based on Pearson correlation with a target category.

        Computes Pearson correlation between each term's count and a binary target
        vector representing target_class presence. Keeps terms with correlation >= threshold.

        Args:
            target_class: Category name string to correlate against.
            threshold: Minimum Pearson correlation required to keep a term (default 0.1).

        Returns:
            Dict containing result_id, terms_kept, terms_removed, kept_terms preview,
            and removed_terms preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Call build_dtm_tool first."}

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None or len(labels) == 0:
            return {"error": "No labels found in session. Ensure dataset is loaded with category labels."}

        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        if not isinstance(threshold, (int, float)):
            return {"error": "threshold must be a float."}

        # Target binary vector
        y = (np.asarray(labels) == target_class).astype(float)
        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' has zero variance (present in all or no documents)."}

        # Calculate Pearson correlation per term
        X = session.feature_matrix.toarray()
        y_centered = y - np.mean(y)
        y_std = np.std(y)

        X_centered = X - np.mean(X, axis=0)
        X_std = np.std(X, axis=0)

        # Guard against zero variance columns
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.dot(y_centered, X_centered) / (len(y) * y_std * X_std)
            correlations = np.nan_to_num(correlations, nan=0.0)

        keep_mask = correlations >= threshold
        kept_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        if len(kept_indices) == 0:
            return {
                "error": f"Filter threshold {threshold} removes all terms. DTM was not modified."
            }

        original_feature_names = list(session.feature_names)
        kept_terms = [original_feature_names[i] for i in kept_indices]
        removed_terms = [original_feature_names[i] for i in removed_indices]

        # Store report DataFrame using full original term list BEFORE modifying session
        report_df = pd.DataFrame(
            {"term": original_feature_names, "pearson_r": correlations}
        ).sort_values(by="pearson_r", ascending=False)

        # Update session DTM
        session.feature_matrix = session.feature_matrix[:, kept_indices]
        session.feature_names = kept_terms

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "terms_kept": len(kept_terms),
            "terms_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
            "threshold": float(threshold),
            "target_class": target_class,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str, threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Filters DTM terms based on Spearman rank correlation with a target category.

        Computes Spearman correlation between each term's count and a binary target
        vector representing target_class presence. Keeps terms with correlation >= threshold.

        Args:
            target_class: Category name string to correlate against.
            threshold: Minimum Spearman correlation required to keep a term (default 0.1).

        Returns:
            Dict containing result_id, terms_kept, terms_removed, kept_terms preview,
            and removed_terms preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Call build_dtm_tool first."}

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None or len(labels) == 0:
            return {"error": "No labels found in session. Ensure dataset is loaded with category labels."}

        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        if not isinstance(threshold, (int, float)):
            return {"error": "threshold must be a float."}

        # Target binary vector
        y = (np.asarray(labels) == target_class).astype(float)
        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' has zero variance."}

        X = session.feature_matrix.toarray()
        correlations = np.zeros(X.shape[1])

        for col_idx in range(X.shape[1]):
            r, _ = spearmanr(X[:, col_idx], y)
            correlations[col_idx] = 0.0 if np.isnan(r) else r

        keep_mask = correlations >= threshold
        kept_indices = np.where(keep_mask)[0]
        removed_indices = np.where(~keep_mask)[0]

        if len(kept_indices) == 0:
            return {
                "error": f"Filter threshold {threshold} removes all terms. DTM was not modified."
            }

        original_feature_names = list(session.feature_names)
        kept_terms = [original_feature_names[i] for i in kept_indices]
        removed_terms = [original_feature_names[i] for i in removed_indices]

        # Store report DataFrame using full original term list BEFORE modifying session
        report_df = pd.DataFrame(
            {"term": original_feature_names, "spearman_r": correlations}
        ).sort_values(by="spearman_r", ascending=False)

        # Update session DTM
        session.feature_matrix = session.feature_matrix[:, kept_indices]
        session.feature_names = kept_terms

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "terms_kept": len(kept_terms),
            "terms_removed": len(removed_terms),
            "kept_terms": kept_terms[:20],
            "removed_terms": removed_terms[:20],
            "threshold": float(threshold),
            "target_class": target_class,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    return [
        variance_filter_tool,
        pearson_filter_tool,
        spearman_filter_tool,
    ]
