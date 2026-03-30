import sys
from pathlib import Path

from program.config import AgentWRetreiverConfig
from program.program import AgentWRetreiver
from program.retriever import ExampleRetriever

if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    config = AgentWRetreiverConfig(num_fetch=1)
    retriever = ExampleRetriever(config, needed_param="hi")
    agent = AgentWRetreiver(config, retriever=retriever)
    repo_path = f"{username}/nested_repo_2"
    extra = str(Path(__file__).resolve().parent.parent.parent / "extra_files" / "extra.yaml")
    agent.push_to_hub(repo_path, with_code=True, extra_files=[extra])
