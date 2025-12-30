# Docs Hosting

This project uses MkDocs with the Material theme.

## Local Build

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Build output:

```bash
mkdocs build
```

## GitHub Pages (Example)

1) Add a GitHub Actions workflow that runs `mkdocs build`.
2) Publish the `site/` directory to Pages.

If you want a ready-to-run workflow, create a `.github/workflows/docs.yml`
that installs Python, runs MkDocs, and publishes `site/`.
