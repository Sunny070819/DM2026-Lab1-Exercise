import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_data import make_tools as make_data_tools


def _load_fixture_session() -> tuple[SessionState, dict]:
    """Helper to initialize SessionState and load sample_fixture.csv."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    assert "error" not in load_res
    
    dtm_tools = {t.name: t for t in make_dtm_tools(session)}
    return session, dtm_tools


def test_build_dtm_tool_fixture_4():
    session, dtm_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]

    # Verify state before
    df_before = session.dataframe.copy()
    labels_before = session.labels.copy() if session.labels is not None else None

    res = build_dtm.invoke({})
    assert "error" not in res

    # Verify summary fields match fixture #4
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    assert res["vocabulary_preview"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.artifacts.get("count_vectorizer") is not None

    # Read-only check for dataframe and labels
    assert session.dataframe.equals(df_before)
    if labels_before is not None:
        assert (session.labels == labels_before).all()


def test_build_dtm_tool_errors():
    # 1. Calling build_dtm before dataset load
    empty_session = SessionState()
    dtm_tools_empty = {t.name: t for t in make_dtm_tools(empty_session)}
    res1 = dtm_tools_empty["build_dtm_tool"].invoke({})
    assert "error" in res1

    # 2. Non-existent column
    session, dtm_tools = _load_fixture_session()
    res2 = dtm_tools["build_dtm_tool"].invoke({"column_name": "non_existent_col"})
    assert "error" in res2

    # 3. Invalid ngram_range
    res3 = dtm_tools["build_dtm_tool"].invoke({"ngram_range": [2, 1]})
    assert "error" in res3

    # 4. Empty text column / vocabulary
    session.dataframe["empty_col"] = ""
    res4 = dtm_tools["build_dtm_tool"].invoke({"column_name": "empty_col"})
    assert "error" in res4


def test_term_frequency_tool_fixture_14():
    session, dtm_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]
    term_freq = dtm_tools["term_frequency_tool"]

    # Test error when called before DTM is built
    err_res = term_freq.invoke({})
    assert "error" in err_res

    # Build DTM
    build_dtm.invoke({})

    # Capture feature_matrix state before
    matrix_before = session.feature_matrix.toarray().copy()

    # Call term_frequency_tool
    res = term_freq.invoke({})
    assert "error" not in res

    assert res["total_terms"] == 5
    expected_frequencies = [
        {"term": "always", "count": 7},
        {"term": "gamma", "count": 7},
        {"term": "alpha", "count": 6},
        {"term": "delta", "count": 4},
        {"term": "beta", "count": 3},
    ]
    assert res["term_frequencies"] == expected_frequencies

    # Test top_n parameter validation
    err_top_n = term_freq.invoke({"top_n": 0})
    assert "error" in err_top_n

    # Verify session.feature_matrix remained unchanged
    assert (session.feature_matrix.toarray() == matrix_before).all()


def test_dtm_heatmap_tool_fixture_21():
    session, dtm_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]
    heatmap_tool = dtm_tools["dtm_heatmap_tool"]

    # Test error when called before DTM is built
    err_res = heatmap_tool.invoke({})
    assert "error" in err_res

    build_dtm.invoke({})

    # Test default heatmap call (8x5 full matrix)
    res_default = heatmap_tool.invoke({})
    assert "error" not in res_default
    assert res_default["n_documents_rendered"] == 8
    assert res_default["n_terms_rendered"] == 5
    assert res_default["matrix"][0] == [3, 1, 1, 0, 0]
    assert res_default["matrix"][7] == [0, 0, 0, 0, 0]
    assert session.pending_figure is not None

    # Reset pending_figure to verify subsequent call sets it
    session.pending_figure = None

    # Test sliced call n_terms=2, n_documents=3
    res_sliced = heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert "error" not in res_sliced
    assert res_sliced["n_documents_rendered"] == 3
    assert res_sliced["n_terms_rendered"] == 2
    assert res_sliced["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None

    # Test invalid parameters
    err_terms = heatmap_tool.invoke({"n_terms": 0})
    assert "error" in err_terms

    err_docs = heatmap_tool.invoke({"n_documents": -1})
    assert "error" in err_docs


def test_state_reset_on_sample_data():
    session, dtm_tools = _load_fixture_session()
    dtm_tools["build_dtm_tool"].invoke({})

    assert session.feature_matrix is not None
    assert session.feature_names is not None

    # Instantiate tools_data and sample dataset
    data_tools = {t.name: t for t in make_data_tools(session)}
    sample_tool = data_tools["sample_data_tool"]

    sample_res = sample_tool.invoke({"n": 4, "random_state": 42})
    assert "error" not in sample_res

    # Verify DTM feature_matrix and feature_names are reset to None
    assert session.feature_matrix is None
    assert session.feature_names is None
