import pytest
import pandas as pd

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _load_fixture_session():
    """Helper function to set up session state and load sample_fixture.csv dataset."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_zero_vocab_filtering():
    """Test fixture #17: variance and term_frequency result in zero vocabulary filtering."""
    session, mine_patterns_tool = _load_fixture_session()

    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert "error" not in res
            assert res["pattern_count"] == 0
            assert res["patterns"] == []
            assert "note" in res
            assert res["note"] == "no terms survived filtering"

            # Check result stored in session results_store
            res_id = res["result_id"]
            stored = session.results_store[res_id]
            assert stored["full_report"] is None


def test_mine_patterns_fpgrowth_fixture_17():
    """Test fixture #17: tfidf filtering with fpgrowth algorithm."""
    session, mine_patterns_tool = _load_fixture_session()

    # catA with min_sup=1 to 3 should find alpha with support 3
    for min_sup in [1, 2, 3]:
        res = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup
        })
        assert res["pattern_count"] == 1
        assert len(res["patterns"]) == 1
        assert "alpha" in res["patterns"][0]["pattern"]
        assert res["patterns"][0]["support"] == 3

        # Check full_report
        res_id = res["result_id"]
        full_report = session.results_store[res_id]["full_report"]
        assert isinstance(full_report, pd.DataFrame)
        assert list(full_report.columns) == ["pattern", "support"]
        assert len(full_report) == 1
        assert full_report.iloc[0]["support"] == 3

    # catA with min_sup=4 should return 0 patterns
    res_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_sup4["pattern_count"] == 0
    assert res_sup4["patterns"] == []

    # catB with min_sup=1 to 4 should find gamma with support 4
    for min_sup in [1, 2, 3, 4]:
        res = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup
        })
        assert res["pattern_count"] == 1
        assert len(res["patterns"]) == 1
        assert "gamma" in res["patterns"][0]["pattern"]
        assert res["patterns"][0]["support"] == 4

        res_id = res["result_id"]
        full_report = session.results_store[res_id]["full_report"]
        assert isinstance(full_report, pd.DataFrame)
        assert list(full_report.columns) == ["pattern", "support"]
        assert len(full_report) == 1
        assert full_report.iloc[0]["support"] == 4


def test_mine_patterns_topk_fixture_20():
    """Test fixture #20: tfidf filtering with topk algorithm."""
    session, mine_patterns_tool = _load_fixture_session()

    for k in [1, 2, 3]:
        # catA -> alpha (support 3)
        res_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_a["pattern_count"] == 1
        assert "alpha" in res_a["patterns"][0]["pattern"]
        assert res_a["patterns"][0]["support"] == 3

        # catB -> gamma (support 4)
        res_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_b["pattern_count"] == 1
        assert "gamma" in res_b["patterns"][0]["pattern"]
        assert res_b["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth_fixture_20():
    """Test fixture #20: tfidf filtering with maxfpgrowth algorithm returns 0 patterns."""
    session, mine_patterns_tool = _load_fixture_session()

    for cat in ["catA", "catB"]:
        for min_sup in [1, 4]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup
            })
            assert res["pattern_count"] == 0
            assert res["patterns"] == []


def test_mine_patterns_errors():
    """Test validation errors for invalid arguments or missing data."""
    session, mine_patterns_tool = _load_fixture_session()

    # Invalid filtering_method
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "invalid_method",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res

    # Invalid algorithm
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "invalid_algo",
        "min_sup": 1
    })
    assert "error" in res

    # Missing min_sup for fpgrowth
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in res

    # Missing k for topk
    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res

    # Non-existent category_name
    res = mine_patterns_tool.invoke({
        "category_name": "non_existent_cat",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res
