import sys

from multi_module_repo.agent import AgentWRetreiver, AgentWRetreiverConfig
from multi_module_repo.retriever import ExampleRetriever

if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    config = AgentWRetreiverConfig(num_fetch=1)
    retriever = ExampleRetriever(config, needed_param="hi")
    agent = AgentWRetreiver(config, retriever=retriever)
    repo_path = f"{username}/multi_module_repo"
    agent.push_to_hub(repo_path, with_code=True)
