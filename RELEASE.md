# Releasing MathSlate

This checklist keeps the three public delivery channels aligned: the package
on PyPI, the tagged GitHub source, and the JupyterLite preview on GitHub Pages.
AI is an optional extra; a release must continue to work with only the core
dependencies installed.

## Before tagging

1. Start with a clean working tree. Do not include notebook execution output or
   local build folders unless it is intentionally part of the release.
2. Set `__version__` in `mathslate/__init__.py`; `pyproject.toml` reads it
   from there.
3. Run the complete regression suite:

   ```bash
   python -m pytest -q
   ```

4. Build and inspect the distribution:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check dist/*
   ```

5. Keep assistant verification separate from core verification. Install
   `mathslate[ai-gemini]` or `mathslate[starter]` only when testing the optional
   assistant; the tag workflow independently performs a clean core install and
   rejects any AI, keyring, or notebook dependency pulled in by it.

## Publish

1. Commit and push the release preparation, and merge it into `master`.
2. From an up-to-date `master` checkout, tag the version the checkout carries
   and push the tag. `scripts/release_version.py` reads it from
   `mathslate/__init__.py` in the checkout rather than importing `mathslate`,
   which could find a different installed copy:

   ```bash
   git switch master && git pull
   version="$(python scripts/release_version.py)"
   git tag -a "v$version" -m "Release v$version"
   git push origin "v$version"
   ```

   The publish workflow checks the tag against the same file first, and stops
   before testing or uploading anything if they disagree.

3. In GitHub Actions, wait for **Publish distribution to PyPI** to finish its
   full test suite and build. Approve the `pypi` environment when requested.
4. Confirm the version can be installed from PyPI in a fresh virtual
   environment.
5. Create and publish the matching GitHub Release from that tag. This starts
   **Deploy JupyterLite to GitHub Pages**; a tag alone does not update Pages.
6. Verify the published preview at https://berd2.github.io/mathslate/ and run
   one 2D and one 3D plot there.

## If something fails

- Do not overwrite or reuse a PyPI version. Make a new patch version instead.
- A failed Pages deployment can be rerun from **Actions → Deploy JupyterLite to
  GitHub Pages** after fixing the source or release configuration.
- Record the failure and the remediation in the GitHub Release notes or issue
  tracker so the next release has a clear history.
