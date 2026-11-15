# Contributing

Thanks for looking. Bug reports with a screenshot of the preview, ideas from real printing situations, and pull requests are all welcome.

## Setup

```bash
git clone https://github.com/flawnull/a4print.git
```

```bash
cd a4print
```

```bash
pip install -e ".[dev,heic]"
```

Add `max` to the extras to work on maximum quality mode (it pulls in PyTorch).

## Before a pull request

```bash
pytest -q
```

```bash
ruff check a4print tests
```

Both run in CI on Linux, macOS and Windows.

## How changes are judged

a4print changes how images print, so a change to processing should come with evidence: a before/after crop at print size, and a number where one makes sense (sharpness, tone drift, time, memory). [How it works](docs/how-it-works.md) shows the kind of measurements used so far.

- Tests use generated or public-domain images only. Don't add copyrighted pictures, even small ones.
- Keep processing faithful: no changes to faces, colors or composition beyond what the image needs to print well.
- New downloads (models, programs) need a pinned SHA-256 and a license that allows redistribution or download, credited in `THIRD_PARTY_NOTICES.md`.

## Publishing (maintainers)

PyPI publishing uses trusted publishing, so no token is stored anywhere:

1. On pypi.org, add a trusted publisher for project `a4print`: owner `flawnull`, repository `a4print`, workflow `publish.yml`, environment `pypi`.
2. Bump the version in `pyproject.toml` and `a4print/__init__.py`, update `CHANGELOG.md`.
3. Run **publish to PyPI** from the Actions tab.
