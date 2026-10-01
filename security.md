# Security Policy

## Supported versions

| Version | Supported | Notes |
| ------- | --------- | ----- |
| 1.3.0+ | :white_check_mark: | Current; includes subprocess + GUI hardening |
| 1.2.1 | :x: | Older; lacks security protections described below |
| < 1.2.1 | :x: | Not supported |

## What's new in v.1.3.0

This release hardens the codebase against common security issues:

- **Safe subprocess execution** for ORCA: all ORCA invocation now uses argument lists with `shell=False`
- **GUI path validation**: dataset saves are guarded against accidental writes outside the project root
- **Atomic writes**: JSON/CSV edits are written via temporary files to avoid corruption on interruption
- **Security smoke check**: a lightweight source-level check prevents unsafe subprocess patterns from being reintroduced

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, report security issues privately to the project maintainers. Please include:
- a clear description of the issue
- affected files or code paths
- reproduction steps (if applicable)
- impact assessment
- any suggested fix or mitigation

We aim to acknowledge reports promptly and provide updates as we investigate.

## Security expectations for contributors

This project enforces the following practices:

### Subprocess execution
- Always use `subprocess.run([...], shell=False, ...)`
- Never use `shell=True`, `os.system(...)`, or string interpolation for shell commands
- Resolve executables from environment variables or `PATH`, never hardcode local paths

### File operations
- Validate file paths before writing to ensure they remain within the project
- Use atomic writes (temporary file + `os.replace(...)`) for user-edited data
- Avoid silent writes to external directories; surface warnings instead
- Keep generated outputs under `data_json/`, `orca_outputs/`, or `plots/` by default

### Code patterns
The following patterns are **prohibited** in project code:
- `subprocess.run(..., shell=True)`
- `os.system(...)`
- `eval(...)`
- `exec(...)`

Any exception requires explicit justification and review.

### Local development
- Do not commit credentials, tokens, or secrets to the repository
- Use `ORCA_EXE` environment variable instead of hardcoding ORCA paths
- Keep `.env` and local configuration files in `.gitignore`

## Security review expectations

Code changes that introduce:
- subprocess or external command execution
- file import/export logic
- path-based filesystem access
- environment-variable handling

must be reviewed for:
- command injection vulnerabilities
- path traversal issues
- accidental data loss or clobbering
- permission and access control

## Limitations

This policy describes the current protections and minimum required practices. It is not a formal security certification and does not guarantee invulnerability against all classes of attack.

The project is protected against the most common code-level issues in scientific computing and desktop tools, but ongoing review is necessary as the codebase evolves.

## Summary of v1.3.0 baseline

:white_check_mark: ORCA execution is subprocess-safe  
:white_check_mark: GUI dataset writes are path-validated  
:white_check_mark: Atomic writes reduce data corruption risk  
:white_check_mark: Source-level regression checks for unsafe patterns  
:white_check_mark: Clear contributor security expectations