import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def make_tools(session):
    """Factory function to instantiate tools with access to the shared session."""

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: Optional[float] = None,
        n_neighbors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions (PCA, t-SNE, UMAP) and sets a scatter plot on pending_figure.

        Args:
            method: Reduction algorithm ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Target dimensions, default 2.
            random_state: Seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).

        Returns:
            Dict containing reduced coordinates, method, and explained variance ratio (for PCA).
        """
        # Validate DTM
        if session.feature_matrix is None:
            return {"error": "No feature_matrix (DTM) found in session. Please run build_dtm_tool first."}

        method = method.lower()
        valid_methods = ["pca", "tsne", "umap"]
        if method not in valid_methods:
            return {"error": f"Invalid method: '{method}'. Must be one of {valid_methods}."}

        if n_components <= 0:
            return {"error": f"n_components must be positive, got {n_components}."}

        dtm_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.array(session.feature_matrix)
        n_samples = dtm_dense.shape[0]

        if n_samples == 0:
            return {"error": "Document-term matrix has zero documents."}

        # 1. Dimensionality Reduction with try/except
        explained_variance_ratio = None

        try:
            if method == "pca":
                pca = PCA(n_components=n_components, random_state=random_state)
                coords = pca.fit_transform(dtm_dense)
                explained_variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]

            elif method == "tsne":
                perp = perplexity if perplexity is not None else 30.0
                if perp >= n_samples:
                    return {"error": f"t-SNE perplexity ({perp}) must be less than the number of samples ({n_samples})."}

                # Default init to match known answers in TEST_FIXTURE.md #8
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perp,
                    random_state=random_state,
                )
                coords = tsne.fit_transform(dtm_dense)

            elif method == "umap":
                n_neigh = n_neighbors if n_neighbors is not None else 15
                if n_neigh >= n_samples:
                    return {"error": f"UMAP n_neighbors ({n_neigh}) must be less than the number of samples ({n_samples})."}

                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neigh,
                    random_state=random_state,
                )
                coords = reducer.fit_transform(dtm_dense)
        except Exception as e:
            return {"error": f"Dimensionality reduction failed using {method}: {str(e)}"}

        coords_rounded = np.round(coords, 4).tolist()

        # 2. Scatter Plot Generation
        categories = (
            session.dataframe["category_name"].tolist()
            if session.dataframe is not None and "category_name" in session.dataframe.columns
            else [f"Doc {i}" for i in range(n_samples)]
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else np.zeros(n_samples),
            "Category": categories,
        })

        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax,
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # 3. Summary preview (max 50 coordinates) and Full Report
        preview_coords = coords_rounded[:50]
        truncated = len(coords_rounded) > 50

        args_recorded = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
        }

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": preview_coords,
            "truncated": truncated,
            "total_documents": len(coords_rounded),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        coords_df = pd.DataFrame(
            coords_rounded,
            columns=[f"Dim_{i+1}" for i in range(n_components)],
            index=pd.Index(categories, name="category_name"),
        )

        session.artifacts[f"reduced_coords_{method}"] = coords_rounded
        session.store_result("reduce_dimensions_tool", args_recorded, summary, full_report=coords_df)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes (one-hot encodes) category labels from session.dataframe.

        Stores the binary indicator matrix in session.artifacts['binary_labels']
        and returns a summary with category classes and shape.

        Returns:
            Dict containing status, categories list, and binarized matrix shape.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' missing from session.dataframe."}

        categories_series = session.dataframe["category_name"]
        if categories_series.empty:
            return {"error": "DataFrame 'category_name' column is empty."}

        # One-hot encode category labels using pandas get_dummies
        binary_df = pd.get_dummies(categories_series, dtype=int)
        categories_list = binary_df.columns.tolist()

        args_recorded = {}
        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "categories": categories_list,
            "num_samples": len(binary_df),
            "num_classes": len(categories_list),
        }

        session.artifacts["binary_labels"] = binary_df.values.tolist()
        session.store_result("binarize_labels_tool", args_recorded, summary, full_report=binary_df)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
