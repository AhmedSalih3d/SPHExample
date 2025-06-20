# Project Agents.md Guide for OpenAI Codex

This file describes how OpenAI Codex and other AI agents should work with this repository.
It may evolve over time, in shaa Allah.

## Project Structure for OpenAI Codex Navigation
- `/src`: main Julia source code for the SPH solver
- `/example`: example scripts demonstrating solver usage
- `/input`: sample input files used by the examples
- `/images`: images referenced by `README.md`
- `Project.toml`/`Manifest.toml`: dependency declarations
  (do not modify without instruction)
- `README.md`: high level project overview and instructions

## Coding Conventions for OpenAI Codex
- Use Julia for all new code
- Indent with four spaces
- Keep lines under 92 characters where practical
- Use snake_case for functions and variables and CamelCase for types
- Document public functions with docstrings and comment complex logic

## Documentation Standards for OpenAI Codex
- Update `README.md` or example docs when behaviour changes
- Keep explanations concise and clear

## Testing Requirements for OpenAI Codex
Run the test suite before opening a pull request:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
The repository currently lacks a `test` directory, so this command will report an error.
Mention this in the PR until tests are added.

## Pull Request Guidelines for OpenAI Codex
- Reference related issues when applicable
- Keep changes focused on a single concern
- List commands you executed (tests, scripts) in the PR description
- Follow commit message conventions: short imperative summary (≤50 chars)
  Provide details in the body if needed
- Do not amend or rebase pushed commits

## Programmatic Checks for OpenAI Codex
Before submitting a PR run:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
If dependencies changed, also run:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Evolution of Agents.md
These instructions may change as the project grows.
Feel free to open an issue or PR proposing improvements.
