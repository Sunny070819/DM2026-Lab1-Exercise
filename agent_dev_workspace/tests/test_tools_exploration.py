import pytest
import numpy as np
import pandas as pd

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools
from tools_dtm import make_tools as make_dtm_tools


def _load_fixture_session():
    """Helper function to set up session state, load sample_fixture.csv, and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    return session, cosine_similarity_tool, feature_correlation_matrix_tool


def test_cosine_similarity_tool_fixture_16():
    """Test cosine_similarity_tool against known answers in TEST_FIXTURE.md #16."""
    session, cosine_tool, _ = _load_fixture_session()

    # Doc 0 vs 1 (both catA, near-identical text): ~0.9847
    res_0_1 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert "error" not in res_0_1
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 3 (catA vs catB, share only "always"): ~0.0909
    res_0_3 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert "error" not in res_0_3
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Doc 3 vs 4 (both catB, near-identical text): ~0.9847
    res_3_4 = cosine_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert "error" not in res_3_4
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 7 (catA vs empty-text row): 0.0, not NaN
    res_0_7 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert "error" not in res_0_7
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool_fixture_19():
    """Test feature_correlation_matrix_tool against known answers in TEST_FIXTURE.md #19."""
    session, _, corr_tool = _load_fixture_session()

    res = corr_tool.invoke({"top_n": 20})
    assert "error" not in res

    top_terms = res["top_terms"]
    corr_matrix = np.array(res["correlation_matrix"])

    # 5 terms total in sample_fixture DTM
    assert len(top_terms) == 5

    # Check diagonal is 1.0
    diag = np.diag(corr_matrix)
    assert np.allclose(diag, 1.0, atol=1e-3)

    # Convert to DataFrame for term lookup
    corr_df = pd.DataFrame(corr_matrix, index=top_terms, columns=top_terms)

    # Check alpha vs beta correlation: ~0.8885
    assert corr_df.loc["alpha", "beta"] == pytest.approx(0.8885, abs=1e-3)

    # Check gamma vs delta correlation: ~0.8307
    assert corr_df.loc["gamma", "delta"] == pytest.approx(0.8307, abs=1e-3)

    # Check pending_figure is set properly
    assert session.pending_figure is not None

    # Check full_report in results_store
    res_id = res["result_id"]
    full_report = session.results_store[res_id]["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert full_report.loc["alpha", "beta"] == pytest.approx(0.8885, abs=1e-3)


def test_exploration_tools_errors():
    """Test validation errors for missing DTM, invalid indices, and invalid top_n."""
    # Test missing DTM
    session_no_dtm = SessionState()
    tools_no_dtm = make_tools(session_no_dtm)
    cosine_no_dtm = next(t for t in tools_no_dtm if t.name == "cosine_similarity_tool")
    corr_no_dtm = next(t for t in tools_no_dtm if t.name == "feature_correlation_matrix_tool")

    assert "error" in cosine_no_dtm.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert "error" in corr_no_dtm.invoke({"top_n": 20})

    # Test with DTM loaded
    session, cosine_tool, corr_tool = _load_fixture_session()

    # Out-of-range doc indices
    assert "error" in cosine_tool.invoke({"doc_idx1": -1, "doc_idx2": 0})
    assert "error" in cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 999})

    # top_n <= 0
    assert "error" in corr_tool.invoke({"top_n": 0})
    assert "error" in corr_tool.invoke({"top_n": -5})
