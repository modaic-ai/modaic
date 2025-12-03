import sys

from program.program import ExampleProgram, ExampleConfig

if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    agent = ExampleProgram(ExampleConfig(output_type="str"), runtime_param="hi")
    repo_path = f"{username}/simple_repo_with_compile"
    agent.push_to_hub(repo_path, with_code=True)
