import sys

from program.program import ExampleConfig, ExampleProgram

if __name__ == "__main__":
    username = sys.argv[1]  # ← first arg after script name (username)
    program = ExampleProgram(ExampleConfig(output_type="str"), runtime_param="hi")
    repo_path = f"{username}/simple_repo_with_compile"
    program.push_to_hub(repo_path, with_code=True)
