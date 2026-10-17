import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def _load_fixture_session() -> tuple[SessionState, dict, dict]:
    """Helper to initialize SessionState, load sample_fixture.csv, and load tools."""
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
    filtering_tools = {t.name: t for t in make_filtering_tools(session)}

    return session, dtm_tools, filtering_tools


def test_variance_filter_tool_fixture_5():
    """Test variance_filter_tool against known answer #5 in TEST_FIXTURE.md."""
    session, dtm_tools, filtering_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]
    variance_filter = filtering_tools["variance_filter_tool"]

    # (a) Error when DTM is not built
    err_dtm = variance_filter.invoke({"threshold": 0.15})
    assert "error" in err_dtm

    # Build DTM (8 documents, 5 terms: alpha, always, beta, delta, gamma)
    build_dtm.invoke({})

    # Capture state before for (c) verification
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    # (c) Error when threshold removes all terms (e.g. threshold = 2.0)
    err_all_removed = variance_filter.invoke({"threshold": 2.0})
    assert "error" in err_all_removed
    assert (session.feature_matrix.toarray() == matrix_before.toarray()).all()
    assert session.feature_names == names_before

    # Valid filter run with threshold = 0.15
    res = variance_filter.invoke({"threshold": 0.15})
    assert "error" not in res
    assert res["terms_kept"] == 4
    assert res["terms_removed"] == 1
    assert res["kept_terms"] == ["alpha", "beta", "delta", "gamma"]
    assert res["removed_terms"] == ["always"]

    # Verify full report stored in results_store using result_id
    result_id = res["result_id"]
    report_df = session.results_store[result_id]["full_report"]

    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    for term, exp_var in expected_variances.items():
        actual_var = report_df.loc[report_df["term"] == term, "variance"].values[0]
        assert actual_var == pytest.approx(exp_var, abs=1e-3)


def test_pearson_filter_tool_fixture_6():
    """Test pearson_filter_tool against known answer #6 in TEST_FIXTURE.md."""
    session, dtm_tools, filtering_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]
    pearson_filter = filtering_tools["pearson_filter_tool"]

    # (a) Error when DTM is not built
    err_dtm = pearson_filter.invoke({"target_class": "catB", "threshold": 0.0})
    assert "error" in err_dtm

    build_dtm.invoke({})

    # (b) Error when target_class does not exist
    err_class = pearson_filter.invoke({"target_class": "non_existent_class", "threshold": 0.0})
    assert "error" in err_class

    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    # (c) Error when threshold removes all terms
    err_all_removed = pearson_filter.invoke({"target_class": "catB", "threshold": 2.0})
    assert "error" in err_all_removed
    assert (session.feature_matrix.toarray() == matrix_before.toarray()).all()
    assert session.feature_names == names_before

    # (d) Pearson with threshold 0.0: keeps always, delta, gamma; removes alpha, beta
    res = pearson_filter.invoke({"target_class": "catB", "threshold": 0.0})
    assert "error" not in res
    assert res["terms_kept"] == 3
    assert res["terms_removed"] == 2
    assert set(res["kept_terms"]) == {"always", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"alpha", "beta"}

    # Verify full report stored in results_store using result_id
    result_id = res["result_id"]
    report_df = session.results_store[result_id]["full_report"]

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson.items():
        actual_r = report_df.loc[report_df["term"] == term, "pearson_r"].values[0]
        assert actual_r == pytest.approx(exp_r, abs=1e-3)


def test_spearman_filter_tool_fixture_6():
    """Test spearman_filter_tool against known answer #6 in TEST_FIXTURE.md."""
    session, dtm_tools, filtering_tools = _load_fixture_session()
    build_dtm = dtm_tools["build_dtm_tool"]
    spearman_filter = filtering_tools["spearman_filter_tool"]

    # (a) Error when DTM is not built
    err_dtm = spearman_filter.invoke({"target_class": "catB", "threshold": 0.0})
    assert "error" in err_dtm

    build_dtm.invoke({})

    # (b) Error when target_class does not exist
    err_class = spearman_filter.invoke({"target_class": "non_existent_class", "threshold": 0.0})
    assert "error" in err_class

    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    # (c) Error when threshold removes all terms
    err_all_removed = spearman_filter.invoke({"target_class": "catB", "threshold": 2.0})
    assert "error" in err_all_removed
    assert (session.feature_matrix.toarray() == matrix_before.toarray()).all()
    assert session.feature_names == names_before

    # (d) Spearman with threshold 0.0: keeps always, delta, gamma; removes alpha, beta
    res = spearman_filter.invoke({"target_class": "catB", "threshold": 0.0})
    assert "error" not in res
    assert res["terms_kept"] == 3
    assert res["terms_removed"] == 2
    assert set(res["kept_terms"]) == {"always", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"alpha", "beta"}

    # Verify full report stored in results_store using result_id
    result_id = res["result_id"]
    report_df = session.results_store[result_id]["full_report"]

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_r in expected_spearman.items():
        actual_r = report_df.loc[report_df["term"] == term, "spearman_r"].values[0]
        assert actual_r == pytest.approx(exp_r, abs=1e-3)
