# Contributing to Modaic
## Finding an Issue to Work On
- Check the [issues](https://github.com/modaic-ai/modaic/issues) page for open issues.
- If you are new to the project, start with the [good first issue](https://github.com/modaic-ai/modaic/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) label.
- Also check the [help wanted](https://github.com/modaic-ai/modaic/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) label for issues that are more complex but still manageable.
- If you have questions about an issue, ask in the comments.
- If you want to work on an issue, comment on the issue that you will work on it.
- If you have already started working on an issue, let us know in the comments.

## Setting up the Development Environment
- First, fork the repository and clone it locally. 
- We use [uv](https://docs.astral.sh/uv/) to manage the dependencies. Refer to the site for installation instructions. 
- Then install the development dependencies with the following command:
```bash
cd modaic
uv sync --dev
```

## Code Style and Formatting
**Comments**

Include docstrings for all public functions and classes.
Do not use # comments unless they are prefixed with a code tag and are necessary for the code to be understandable. We use the following code tags:
- `# CAVEAT:` - A heads-up that there’s something tricky or non-obvious here that the reader should keep in mind.
- `# NOTE:` - A note to the reader offering some necessary context.
- `# TODO:` - Something needs to be done.
- `# DOCME:` - Needs to be documented.
- `# BUG:` - Something is wrong.
- `# FIXME:` - Something needs to be fixed.
- `# HACK:` - A temporary ugly workaround solution that is hacky and should be changed.
To get the most use out of code tags, I recommend you install the [TODO Tree Extension](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree)

**Formatting**

We use the ruff linter/formatter to check for code style and formatting. It is installed with the dev dependencies. To use it, install the [Ruff Extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) on VSCode. Make sure your VSCode python interpreter is set to the python path in the `.venv` created by uv. You can change it by pressing `Ctrl+Shift+P` on windows and `Cmd+Shift+P` on Mac and typing `Python: Select Interpreter`. You should see one named `.venv`. This will ensure the formatter rules match the modaic specific formatting.

When working with ruff, you may find these settings useful for auto-formatting code. You can add these to your VSCode user settings. (cmd + shift + p -> "Preferences: Open Workspace Settings (JSON)")
```json
  "editor.formatOnSave": true,
  "ruff.organizeImports": true,
  "ruff.path": ["ruff"],
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  }
```

## Contributing to the Documentation
- Follow instructions for setting up the local development environment [above](#setting-up-the-development-environment). 
- Next ensure you have [node](https://nodejs.org/en/download/) installed. 
- Install the node dependencies for the documentation.
```bash
npm install
```
- To run the documentation locally with live reload, run the following command:
```bash
npm run dev
```
- To build the documentation, run the following command:
```bash
npm run build
```