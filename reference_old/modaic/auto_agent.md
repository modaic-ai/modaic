# Table of Contents

* [auto\_agent](#modaic.auto_agent)
  * [AutoConfig](#modaic.auto_agent.AutoConfig)
  * [AutoAgent](#modaic.auto_agent.AutoAgent)
    * [from\_precompiled](#modaic.auto_agent.AutoAgent.from_precompiled)
  * [git\_snapshot](#modaic.auto_agent.git_snapshot)

---
sidebar_label: auto_agent
title: modaic.auto_agent
---

## AutoConfig Objects

```python
class AutoConfig()
```

Config for AutoAgent.

## AutoAgent Objects

```python
class AutoAgent()
```

The AutoAgent class used to dynamically load agent frameworks at the given Modaic Hub path

#### from\_precompiled

```python
@staticmethod
def from_precompiled(repo_id, **kw)
```

Load a compiled agent from the given path. AutoAgent will automatically determine the correct Agent class.

**Arguments**:

- `repo_id` - The path to the compiled agent.
- `**kw` - Additional keyword arguments to pass to the Agent class.
  

**Returns**:

  An instance of the Agent class.

#### git\_snapshot

```python
def git_snapshot(url: str, *, rev: str | None = "main") -> str
```

Clone / update a public Git repo into a local cache and return the path.

**Arguments**:

- `url` - Git repository URL (e.g., https://github.com/user/repo.git)
- `rev` - Branch, tag, or full commit SHA; default is &#x27;main&#x27;
  

**Returns**:

  Path to the local cached repository.

