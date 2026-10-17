import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def setup_pipeline_with_dtm():
    """Helper to initialize session with loaded dataset and built DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using tools_dtm's make_tools pattern
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    reduction_tools = make_tools(session)
    return session, reduction_tools

def test_pca_known_answer_7():
    session, reduction_tools = setup_pipeline_with_dtm()
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca", "n_components": 2})
    assert "error" not in res
    
    expected_variance_ratio = [0.7532, 0.1965]
    expected_coordinates = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    
    assert np.allclose(res["explained_variance_ratio"], expected_variance_ratio, atol=1e-3)
    assert np.allclose(res["coordinates"], expected_coordinates, atol=1e-3)
    assert res["total_documents"] == len(session.dataframe)
    assert res["truncated"] is False
    assert session.pending_figure is not None

def test_tsne_known_answer_8():
    session, reduction_tools = setup_pipeline_with_dtm()
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    assert "error" not in res
    
    coords = np.array(res["coordinates"])
    
    # 1. Coordinate shape is (8, 2)
    assert coords.shape == (8, 2)
    
    # 2. All coordinate values are finite numbers
    assert np.all(np.isfinite(coords))
    
    # 3. Coordinates of documents 5 and 6 (zero-based) are identical
    assert np.allclose(coords[5], coords[6])
    
    # 4. Running the tool twice with the same random_state gives identical coordinates
    res2 = reduce_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    assert np.allclose(coords, res2["coordinates"])
    
    # 5. total_documents == 8
    assert res["total_documents"] == 8
    
    # 6. truncated is False
    assert res["truncated"] is False
    
    # 7. session.pending_figure is not None
    assert session.pending_figure is not None

def test_umap_known_answer_9():
    session, reduction_tools = setup_pipeline_with_dtm()
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    assert "error" not in res
    
    expected_coordinates = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]
    
    assert np.allclose(res["coordinates"], expected_coordinates, atol=1e-1)
    assert res["total_documents"] == len(session.dataframe)
    assert res["truncated"] is False
    assert session.pending_figure is not None

def test_reduce_dimensions_error_cases():
    session = SessionState()
    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # Missing DTM
    res_no_dtm = reduce_tool.invoke({"method": "pca"})
    assert "error" in res_no_dtm
    
    # Setup DTM for parameter validation checks
    _, reduction_tools = setup_pipeline_with_dtm()
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    # Invalid method
    res_inv_method = reduce_tool.invoke({"method": "invalid_method"})
    assert "error" in res_inv_method
    
    # n_components <= 0
    res_invalid_ncomp = reduce_tool.invoke({"method": "pca", "n_components": 0})
    assert "error" in res_invalid_ncomp
    
    # perplexity >= n_samples (n_samples is 8)
    res_high_perp = reduce_tool.invoke({"method": "tsne", "perplexity": 8})
    assert "error" in res_high_perp
    
    # n_neighbors >= n_samples (n_samples is 8)
    res_high_neigh = reduce_tool.invoke({"method": "umap", "n_neighbors": 8})
    assert "error" in res_high_neigh

def test_binarize_labels_tool():
    # Missing dataframe error case
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    binarize_tool_empty = next(t for t in tools_empty if t.name == "binarize_labels_tool")
    res_err = binarize_tool_empty.invoke({})
    assert "error" in res_err
    
    # Valid setup with dataset loaded
    session, reduction_tools = setup_pipeline_with_dtm()
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    assert "error" not in res
    
    # Compute expected classes and matrix dynamically from session.dataframe
    binary_df = pd.get_dummies(session.dataframe["category_name"], dtype=int)
    expected_categories = binary_df.columns.tolist()
    expected_binary_matrix = binary_df.values.tolist()
    
    assert res["categories"] == expected_categories
    assert res["num_samples"] == len(session.dataframe)
    assert res["num_classes"] == len(expected_categories)
    assert session.artifacts.get("binary_labels") == expected_binary_matrix
