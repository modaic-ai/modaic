import sys

from simple_repo_with_compile.agent import ExampleAgent, ExampleConfig

if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    agent = ExampleAgent(ExampleConfig(output_type="str"), runtime_param="hi")
    repo_path = f"{username}/simple_repo_with_compile"
    agent.push_to_hub(repo_path, with_code=True)
