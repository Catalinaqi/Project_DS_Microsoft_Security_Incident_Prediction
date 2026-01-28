# Project_DS_Microsoft_Security_Incident_Prediction

This repository is configured to be reproducible across machines using **Python 3.10** and **Poetry**.
The most common problems when cloning a Python repo are: wrong Python version, Poetry using a different virtualenv than the IDE, and “cache” behavior where changes don’t seem to apply. This guide is designed to avoid that.

---

## What is Poetry

**Poetry** is a dependency + environment manager for Python projects (similar in spirit to Maven/Gradle in Java):

* Reads project metadata and dependencies from `pyproject.toml`
* Resolves exact versions into `poetry.lock`
* Creates/uses a virtual environment
* Installs dependencies into that environment
* Optionally installs the project itself as an importable package (so `import crispdm` works everywhere)

---

## Repository layout and imports (important)

This repository uses a **src layout**:

* `src/` contains the Python source code
* `src/crispdm/` and `src/crispml/` are the internal packages
* `notebooks/`, `data/`, `config/` are not Python packages; they are resources

Correct imports in code:

* Use:

    * `from crispdm...`
    * `from crispml...`

Avoid:

* `from src.crispdm...`
* `from src.crispml...`

Reason: `src/` is intended to be a container folder, not part of the import path.

Also:

* Keep `__init__.py` inside `src/crispdm/` and `src/crispml/` (those are packages)
* Do **not** create `src/__init__.py` (it makes `src` a package and encourages `src.*` imports)

---

## Requirements

* Python **3.10.x** installed locally
* Poetry installed

Check installed Python versions (Windows):

* PowerShell:

    * `py -0p`

Check Poetry:

* `poetry --version`

---

## Standard setup (recommended)

Goal: create a local virtual environment **inside the project** and force Poetry to use it.

### Windows PowerShell (recommended)

1. Go to project root

* `cd <path-to-your-repo>`

2. Configure Poetry to create the venv inside the project

* `poetry config virtualenvs.in-project true --local`
* `poetry config virtualenvs.create true --local`

3. Force Poetry to use Python 3.10 (update the path for your machine)

* `poetry env use "C:\Users\<YOU>\AppData\Local\Programs\Python\Python310\python.exe"`

4. Install dependencies

* `poetry install`

5. Verify

* `poetry env info -p`
* `poetry run python -c "import sys; print(sys.version); print(sys.executable)"`

You should see:

* the env path pointing to `<repo>\.venv`
* Python 3.10.x
* python executable inside `.venv\Scripts\python.exe`

6. (Optional) Activate the venv in the terminal (so `python` uses 3.10 without `poetry run`)

* `.\.venv\Scripts\Activate.ps1`
* `python --version`

If activation is blocked by policy:

* `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
* `.\.venv\Scripts\Activate.ps1`

### Git Bash / MINGW64

Install & verify (same Poetry commands):

* `poetry install`
* `poetry run python --version`

Activate venv in Git Bash:

* `source ./.venv/Scripts/activate`
* `python --version`

---

## Jupyter / notebooks

### Option A (simple): run Jupyter via Poetry

* `poetry run jupyter lab`

### Option B (recommended): create a named kernel for the venv

This is useful for IDE notebooks and for avoiding “wrong interpreter” issues.

* `poetry run python -m ipykernel install --user --name env-datascience-01 --display-name "env-datascience-01 (Py3.10)"`

Then select the kernel:

* `env-datascience-01 (Py3.10)`

---

## Two possible project modes (depends on `pyproject.toml`)

Poetry can be used in two ways. Which one you use depends on your current `pyproject.toml`.

### Mode 1: Dependency management only (no package install)

This is when:

* `package-mode = false`

Impact:

* Poetry installs dependencies, but your code is not installed as a package.
* To import internal modules (`crispdm`, `crispml`), you usually need `PYTHONPATH` or notebook `sys.path` adjustments.

Common runtime fix:

* PowerShell:

    * `$env:PYTHONPATH = "$PWD\src"`
* Then:

    * `poetry run python -c "import crispdm; print('ok')"`

### Mode 2: Package mode enabled (recommended for clean imports)

This is when:

* `package-mode = true`
* and packages are declared, for example:

    * `packages = [{ include = "crispdm", from = "src" }, { include = "crispml", from = "src" }]`

Impact:

* `poetry install` also installs your project packages into the venv
* You can import without PYTHONPATH:

    * `import crispdm`
    * `import crispml`

If you want clean imports everywhere (scripts + notebooks + CI), Mode 2 is the best.

---

## Common Poetry commands (and what they mean)

* `poetry install`

    * Installs dependencies from `poetry.lock`
    * Creates/uses the venv
    * If package mode is enabled, installs the current project as well

* `poetry add <pkg>`

    * Adds a dependency and updates lock file

* `poetry remove <pkg>`

    * Removes a dependency and updates lock file

* `poetry lock`

    * Re-resolves dependencies and writes `poetry.lock` (does not install)

* `poetry run <command>`

    * Runs a command inside the Poetry-managed venv without manually activating it

* `poetry env info -p`

    * Shows the path of the venv Poetry is using

* `poetry env list --full-path`

    * Lists all venvs Poetry knows for this project

* `poetry env remove --all`

    * Removes venv(s) associated with the project

---

## Troubleshooting (cache / wrong env / changes not applied)

### 1) “python --version” shows Python 3.13 but Poetry shows 3.10

This means your shell is using system Python, but Poetry is correct.

Fix options:

* Use:

    * `poetry run python ...`
* Or activate:

    * `.\.venv\Scripts\Activate.ps1`

### 2) Poetry uses an env in AppData instead of `.venv`

Force local env creation again:

* `poetry config virtualenvs.in-project true --local`
* `poetry config virtualenvs.create true --local`

Then rebuild:

* `poetry env remove --all`
* delete `.venv` if it still exists
* `poetry install`

### 3) “poetry install” doesn’t pick up my code changes

If you changed project packaging configuration (packages list, structure, etc.), do:

* `poetry install`

If it still behaves weird, do a clean rebuild:

* `poetry env remove --all`
* delete `.venv`
* `poetry env use "<path-to-python310>"`
* `poetry install`

### 4) Windows “Access denied” when deleting `.venv`

Close anything that uses Python:

* IntelliJ / PyCharm
* running Jupyter server
* any python processes

Then retry deletion.

---

## Minimal verification checklist

Run from repo root:

* `poetry env info -p`
* `poetry run python --version`
* `poetry run python -c "import crispdm; print('crispdm ok')"`
* `poetry run python -c "import crispml; print('crispml ok')"`

If all pass, the environment is correctly configured.

---

## IDE notes (IntelliJ / PyCharm)

* Set interpreter to:

    * `<repo>\.venv\Scripts\python.exe`
* If the editor still shows stale import errors:

    * Invalidate caches / restart the IDE
* For notebooks in IDE:

    * choose kernel `env-datascience-01 (Py3.10)` or run Jupyter via `poetry run`

---


