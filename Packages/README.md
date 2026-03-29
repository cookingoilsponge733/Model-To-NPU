# Packages

This folder stores APK artifacts tracked in the repository instead of GitHub Releases.

## Current artifact

- `SDXL-NPU-debug-0.1.0.apk` — debug APK installed via `adb install -r` for iterative phone-side testing.

## Notes

- This is a debug build, so it is suitable for testing and preserves app data when updated over an existing debug install with the same package signature.
- The repository intentionally does not rely on GitHub Releases for this artifact right now.
