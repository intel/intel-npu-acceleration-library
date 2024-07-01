# Developer Guide

Install developer packages by typing

```bash
pip install .[dev]
```

It is suggested to install the package locally by using `pip install -e .[dev]`

## Git hooks

All developers should install the git hooks that are tracked in the `.githooks` directory. We use the pre-commit framework for hook management. The recommended way of installing it is using pip:

```bash
pre-commit install
```

If you want to manually run all pre-commit hooks on a repository, run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

Uninstalling the hooks can be done using

```bash
pre-commit uninstall
```

## Testing the library

### Python test

Python test uses `pytest` library. Type

```bash
cd test/python && pytest
```

to run the full test suite.

## Build the documentation

This project uses `sphinx` to build and deploy the documentation. To serve locally the documentation type

```bash
mkdocs serve
```

to deploy it into github pages type

```bash
cd docs
python build_doc.py gh-deploy
```

## Generate python packages

On windows:

```bat
python setup.py sdist
set CIBW_BUILD=cp*
cibuildwheel --platform windows --output-dir dist
```


## Publishing packets

Install twine
```bat
python3 -m pip install --upgrade twine
```

Then check on the built sdist and wheel that are properly formatted (all files should return a green `PASSED`)

```bat
twine check dist/*
```

Upload the packets to `testpypi`

```bat
twine upload --repository testpypi dist/*
```

To upload them to the real index (**verify first with testpypi**)
```bat
twine upload dist/*
```