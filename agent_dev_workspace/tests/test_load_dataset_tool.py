from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # 1. Initialize session state
    session = SessionState()

    # 2. Load premade tools from 'premade_tools' directory
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 3. Invoke load_dataset_tool with sample fixture
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 4. Assert against Known Answer #1 in TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}

    # Verify session state dataframe is also updated properly
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
