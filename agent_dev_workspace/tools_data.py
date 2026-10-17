from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState) -> List[Any]:
    """Factory function to build data-handling tools bound to the session."""

    @tool
    def check_missing_tool(column_name: str = "text") -> Dict[str, Any]:
        """Checks for missing or empty text values in a DataFrame column.

        Identifies nulls (NaN) and empty/whitespace-only strings in the specified column.

        Args:
            column_name: The column to inspect for missing/empty values (default is 'text').

        Returns:
            Dict containing result_id, column name, total_rows, missing_count, and
            a truncated list of the first 10 missing indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in working DataFrame."}

        # Define missing as NaN OR empty/whitespace-only string
        col = session.dataframe[column_name]
        is_null_or_empty = col.isna() | (col.astype(str).str.strip() == "")

        missing_count = int(is_null_or_empty.sum())
        missing_indices = session.dataframe.index[is_null_or_empty].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "column": column_name,
            "total_rows": len(session.dataframe),
            "missing_count": missing_count,
            "missing_indices": missing_indices[:10],
        }

        session.store_result("check_missing_tool", {"column_name": column_name}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False, subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check for (and optionally drop) duplicate documents in session.dataframe.

        By default, checks for exact whole-row duplicates across all columns in session.dataframe.
        If drop=True, drops every copy of any duplicated row using keep=False, matching
        Master's drop_duplicates strategy.

        Args:
            drop (bool): If True, drops duplicate rows from session.dataframe using keep=False. Defaults to False.
            subset (Optional[List[str]]): List of column names to consider for identifying duplicates.
                If None, considers all columns (whole-row match). Defaults to None.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, duplicate_count, rows_dropped, and rows_remaining.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        df = session.dataframe

        if subset is not None:
            missing_cols = [c for c in subset if c not in df.columns]
            if missing_cols:
                return {"error": f"Column(s) {missing_cols} not found in working DataFrame."}

        initial_total = len(df)

        # duplicate_count measures redundant rows (extra copies beyond the first instance)
        duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())

        if drop:
            # Drop every copy of duplicate rows using keep=False
            df_cleaned = df.drop_duplicates(subset=subset, keep=False)
            rows_dropped = initial_total - len(df_cleaned)

            # Reassign session.dataframe with reset index
            session.dataframe = df_cleaned.reset_index(drop=True)

            # Sync category labels if category_name column exists
            if "category_name" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].values)

            # Invalidate downstream DTM feature matrix if present, since row count changed
            if session.feature_matrix is not None:
                session.feature_matrix = None
                session.feature_names = None

            rows_remaining = len(session.dataframe)
        else:
            rows_dropped = 0
            rows_remaining = initial_total

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "rows_dropped": rows_dropped,
            "rows_remaining": rows_remaining,
        }

        session.store_result(
            "check_duplicates_tool", {"drop": drop, "subset": subset}, summary
        )
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Subsamples the working DataFrame session.dataframe in place.

        Args:
            n (Optional[int]): Number of items/rows to sample.
            frac (Optional[float]): Fraction of axis items to sample (e.g. 0.5 for 50%).
            random_state (Optional[int]): Seed for random number generator. Defaults to 42.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, sampled_indices,
            initial_total_rows, and n_sampled.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        initial_total_rows = len(session.dataframe)
        if initial_total_rows == 0:
            return {"error": "Working DataFrame is empty. Cannot sample."}

        if (n is None and frac is None) or (n is not None and frac is not None):
            return {"error": "Specify either 'n' or 'frac', not both."}

        if n is not None:
            if not isinstance(n, int) or n <= 0 or n > initial_total_rows:
                return {
                    "error": f"n must be a positive integer <= total rows ({initial_total_rows})."
                }

        if frac is not None:
            if not isinstance(frac, (int, float)) or frac <= 0.0 or frac > 1.0:
                return {"error": "frac must be a float between 0 and 1."}

        # Subsample using pandas DataFrame.sample
        df_sampled = session.dataframe.sample(
            n=n, frac=frac, random_state=random_state
        )

        if len(df_sampled) == 0:
            return {"error": "Sampling resulted in 0 rows."}

        # Record original row indices before resetting index
        sampled_indices = df_sampled.index.tolist()

        # Update session.dataframe in place with reset index
        session.dataframe = df_sampled.reset_index(drop=True)

        # Sync session.labels if category_name or category column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Invalidate feature matrix & feature names since row alignment changed
        if session.feature_matrix is not None:
            session.feature_matrix = None
            session.feature_names = None

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_indices": sampled_indices,
            "initial_total_rows": initial_total_rows,
            "n_sampled": len(session.dataframe),
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """Lists files and folders in the repository or a specified subdirectory.

        Args:
            subdirectory (str): Relative path of subdirectory to list. Defaults to root ("").

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, subdirectory,
            directories list, and files list.
        """
        if ".." in subdirectory or Path(subdirectory).is_absolute():
            return {"error": "Invalid subdirectory path. Traversal ('..') and absolute paths are disabled."}

        base_path = Path.cwd()
        target_path = (base_path / subdirectory).resolve()

        if not target_path.exists() or not target_path.is_dir():
            return {"error": f"Subdirectory '{subdirectory}' does not exist or is not a directory."}

        try:
            directories = sorted([p.name for p in target_path.iterdir() if p.is_dir() and not p.name.startswith(".")])
            files = sorted([p.name for p in target_path.iterdir() if p.is_file() and not p.name.startswith(".")])
        except Exception as e:
            return {"error": f"Failed to list directory contents: {str(e)}"}

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def describe_data_tool(column_name: str = "text") -> Dict[str, Any]:
        """Computes text length statistics and generates a boxplot grouped by category.

        Computes character counts for document text, stores 'text_length' in session.dataframe,
        calculates descriptive statistics overall and per category, and sets a boxplot figure
        on session.pending_figure.

        Args:
            column_name (str): Column name containing document text. Defaults to "text".

        Returns:
            Dict[str, Any]: Summary stats dictionary containing result_id, overall stats,
            and stats per category (if category_name exists).
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in working DataFrame."}

        df = session.dataframe

        # Compute text length (character count), empty/null text = 0
        text_lengths = df[column_name].apply(lambda x: len(str(x)) if pd.notna(x) and str(x).strip() != "" else 0)
        df["text_length"] = text_lengths

        overall_desc = df["text_length"].describe().to_dict()

        by_category = {}
        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if category_col is not None:
            grouped = df.groupby(category_col)["text_length"]
            for cat, group in grouped:
                by_category[str(cat)] = group.describe().to_dict()

        # Build boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_col is not None:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {category_col}")
            ax.set_xlabel(category_col)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Character Length")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_desc,
            "by_category": by_category,
        }

        session.store_result("describe_data_tool", {"column_name": column_name}, summary)
        return summary

    @tool
    def tokenize_tool(column_name: str = "text") -> Dict[str, Any]:
        """Tokenizes document text into lowercased unigrams using NLTK.

        Stores list of tokens for each document in session.dataframe['unigrams'] and
        session.artifacts['unigrams'].

        Args:
            column_name (str): Document text column to tokenize. Defaults to "text".

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, method, total_tokens,
            and a preview of tokens for the first 3 documents.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in working DataFrame."}

        def tokenize_text(val: Any) -> List[str]:
            if pd.isna(val) or str(val).strip() == "":
                return []
            return nltk.word_tokenize(str(val).lower())

        tokenized_docs = session.dataframe[column_name].apply(tokenize_text).tolist()

        # Store tokens in session dataframe and artifacts
        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        total_tokens = sum(len(doc) for doc in tokenized_docs)
        preview = tokenized_docs[:3]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "method": "nltk.word_tokenize",
            "total_tokens": total_tokens,
            "preview": preview,
        }

        session.store_result("tokenize_tool", {"column_name": column_name}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """Peeks at the top n_rows of the working DataFrame session.dataframe.

        Returns structural metadata (column list, data types, row counts) and
        a JSON-serializable list of row records with truncated text and list preview.

        Args:
            n_rows (int): Number of rows to inspect. Defaults to 5.

        Returns:
            Dict[str, Any]: Dict containing result_id, total_rows, total_columns,
            columns list, dtypes dict, n_rows_requested, n_rows_returned, and data records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if not isinstance(n_rows, int) or n_rows <= 0:
            return {"error": "n_rows must be a positive integer."}

        df = session.dataframe
        if len(df) == 0:
            return {"error": "Working DataFrame is empty."}

        effective_n_rows = min(n_rows, 20)
        head_df = df.head(effective_n_rows)

        def clean_cell(val: Any) -> Any:
            if isinstance(val, (list, tuple)):
                cleaned_list = [clean_cell(item) for item in val[:20]]
                return cleaned_list
            if pd.isna(val):
                return None
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, str) and len(val) > 100:
                return val[:100] + "..."
            return val

        records = []
        for _, row in head_df.iterrows():
            records.append({col: clean_cell(row[col]) for col in df.columns})

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "n_rows_requested": n_rows,
            "n_rows_returned": len(records),
            "data": records,
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        list_files_tool,
        describe_data_tool,
        tokenize_tool,
        inspect_data_tool,
    ]
