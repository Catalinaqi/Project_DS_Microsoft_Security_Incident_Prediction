**Poetry config**

````md
### Python environment setup (Poetry, Windows)

To create and use a local virtual environment for this project, run:

```bash
poetry config virtualenvs.in-project true --local
poetry env use .venv\Scripts\python.exe
poetry install
````

**What these commands do**

* `poetry config virtualenvs.in-project true --local`
  Tells Poetry to create the virtual environment **inside the project folder** (in `.venv/`)
  and to apply this setting **only to this project** (not globally).

* `poetry env use .venv\Scripts\python.exe`
  Instructs Poetry to use the Python interpreter from `.venv\Scripts\python.exe`.
  If the `.venv` folder does not exist yet, create it first with:

  ```bash
  python -m venv .venv
  ```

* `poetry install`
  Installs all the dependencies defined in `pyproject.toml` (and uses `poetry.lock` if present)
  into the `.venv` environment, so the project is ready to run.

```
```
