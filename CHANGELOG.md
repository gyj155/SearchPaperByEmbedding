# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-20

### Added

- Created `CHANGELOG.md` to document changes.
- Output files are now saved to the `./output` directory.
- The sidebar is now expanded by default in the Streamlit app.
- The default model in the Streamlit app is now "openai".
- Added `.env` loading to `demo.py` for API key management.
- Updated the guideline in `README.md` with a more realistic example.

### Changed

- Changed "Number of results" to a text input for more flexibility.

### Fixed

- Fixed a `KeyError: 'forum_url'` in `search.py` by using safer dictionary access.
- Corrected a connection error in the Streamlit app when `base_url` is empty.
- Resolved an `AttributeError` in `app.py` by properly handling single paper JSON objects.
- Addressed various linter errors in `app.py`.
