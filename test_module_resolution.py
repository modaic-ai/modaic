import modulefinder
import os

if __name__ == "__main__":
    excludes = [
        "langchain_text_splitters",
        "dspy",
        "pydoc_markdown",
        "test_module_resolution",
        "modulefinder",
        "numpy",
    ]

    finder = modulefinder.ModuleFinder(excludes=excludes)
    script_path = os.path.join(os.path.dirname(__file__), "my_module1.py")
    finder.run_script(script_path)
    for name, mod in finder.modules.items():
        print(name)
        # print(name, getattr(mod, "__file__", None))
