from modaic.auto_agent import AutoAgent


def test_load_from_hub():
    loaded_agent = AutoAgent.from_precompiled(
        "https://github.com/modaic-ai/test_auto_model.git"
    )
    assert loaded_agent.config.agent_type == "ExampleAgent"
    assert loaded_agent.config.output_type == "bool"
    assert loaded_agent("Hello") == "Hi, Hello"
