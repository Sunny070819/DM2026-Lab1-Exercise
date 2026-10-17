import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool():
    # Single session instance shared across dataset loading and tool execution
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load sample_fixture.csv dataset into session
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Instantiate workspace tools from tools_data.py using the same session
    data_tools = make_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    # Execute check_missing_tool
    res = check_missing_tool.invoke({})

    # Assert against Known Answer #2 in TEST_FIXTURE.md
    assert res["missing_count"] == 1
    assert res["total_rows"] == 8


def test_check_duplicates_tool_no_drop():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    res = check_duplicates_tool.invoke({"drop": False})

    # Assert against Known Answer #3 in TEST_FIXTURE.md (drop=False)
    assert res["duplicate_count"] == 1
    assert res["rows_dropped"] == 0
    assert res["rows_remaining"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_drop():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    res = check_duplicates_tool.invoke({"drop": True})

    # Assert against Known Answer #3 in TEST_FIXTURE.md (drop=True, keep=False logic)
    assert res["rows_dropped"] == 2
    assert res["rows_remaining"] == 6
    assert len(session.dataframe) == 6
    assert len(session.labels) == 6


def test_sample_data_tool_known_answer():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    sample_data_tool = next(
        t for t in data_tools if t.name == "sample_data_tool"
    )

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Assert against Known Answer #13 in TEST_FIXTURE.md
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["initial_total_rows"] == 8
    assert res["n_sampled"] == 4
    assert len(session.dataframe) == 4
    assert len(session.labels) == 4


def test_sample_data_tool_error_n_out_of_bounds():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    sample_data_tool = next(
        t for t in data_tools if t.name == "sample_data_tool"
    )

    res = sample_data_tool.invoke({"n": 100})

    # Assert error dictionary returned
    assert "error" in res
    assert "n must be a positive integer" in res["error"]


def test_sample_data_tool_frac():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    sample_data_tool = next(
        t for t in data_tools if t.name == "sample_data_tool"
    )

    res = sample_data_tool.invoke({"frac": 0.5, "random_state": 42})

    assert res["initial_total_rows"] == 8
    assert res["n_sampled"] == 4
    assert len(session.dataframe) == 4
    assert len(session.labels) == 4


def test_list_files_tool_valid():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert "error" not in res
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_path_traversal():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": ".."})

    assert "error" in res


def test_describe_data_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    describe_data_tool = next(
        t for t in data_tools if t.name == "describe_data_tool"
    )

    res = describe_data_tool.invoke({})

    assert "error" not in res
    assert session.dataframe["text_length"].tolist() == [29, 23, 17, 30, 24, 18, 18, 0]
    assert session.pending_figure is not None


def test_tokenize_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    tokenize_tool = next(
        t for t in data_tools if t.name == "tokenize_tool"
    )

    res = tokenize_tool.invoke({})

    assert "error" not in res
    assert res["method"] == "nltk.word_tokenize"
    assert session.dataframe["unigrams"].iloc[0] == [
        "always",
        "alpha",
        "alpha",
        "alpha",
        "beta",
    ]
    assert session.dataframe["unigrams"].iloc[7] == []


def test_inspect_data_tool_default():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({})

    assert res["total_rows"] == 8
    assert res["n_rows_returned"] == 5
    assert "result_id" in res
    assert isinstance(res["result_id"], str)
    assert len(res["result_id"]) > 0
    assert res["columns"] == list(session.dataframe.columns)


def test_inspect_data_tool_custom_n_rows():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 20})

    assert res["n_rows_requested"] == 20
    assert res["n_rows_returned"] == 8


def test_inspect_data_tool_missing_value():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 8})

    assert res["data"][7]["text"] == ""


def test_inspect_data_tool_nan_value():
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({"val": [float("nan")]})

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 1})

    assert "error" not in res
    assert res["data"][0]["val"] is None


def test_inspect_data_tool_invalid_n_rows():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 0})

    assert "error" in res
    assert res["error"] == "n_rows must be a positive integer."


def test_inspect_data_tool_no_data_loaded():
    session = SessionState()
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({})

    assert "error" in res
    assert res["error"] == "No dataframe loaded in session. Call load_dataset_tool first."


def test_inspect_data_tool_read_only():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    df_before = session.dataframe.copy(deep=True)
    res = inspect_data_tool.invoke({"n_rows": 5})

    assert "error" not in res
    assert session.dataframe.equals(df_before)


def test_inspect_data_tool_truncation():
    import pandas as pd
    session = SessionState()
    long_text = "a" * 120
    long_list = list(range(25))
    session.dataframe = pd.DataFrame({"long_text": [long_text], "long_list": [long_list]})

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 1})

    assert "error" not in res
    assert len(res["data"][0]["long_text"]) == 103
    assert res["data"][0]["long_text"].endswith("...")
    assert len(res["data"][0]["long_list"]) == 20
