# UV And Submodule Setup Design

## Goal

Convert the OmniDexGrasp setup from conda-oriented instructions to a `uv`-managed workflow, and add the external repositories that the code/configs already reference as Git submodules using SSH URLs.

## Scope

This change covers the local OmniDexGrasp environment and repository wiring:

- Add `pyproject.toml` for `uv sync`.
- Generate and commit `uv.lock` when dependency resolution permits.
- Add Git submodules under `omnidexgrasp/thirdparty/`.
- Update `README.md` installation and usage notes to match the new workflow.
- Keep heavyweight CUDA/source-build dependencies documented explicitly when they cannot be resolved safely through `uv sync`.

This change does not download model checkpoints, datasets, MANO assets, or install system CUDA/toolkit components.

## Submodules

Use SSH URLs for all submodules:

| Path | URL |
| --- | --- |
| `omnidexgrasp/thirdparty/CSDF` | `git@github.com:wrc042/CSDF.git` |
| `omnidexgrasp/thirdparty/EasyHOI` | `git@github.com:lym29/EasyHOI.git` |
| `omnidexgrasp/thirdparty/hamer` | `git@github.com:geopavlakos/hamer.git` |
| `omnidexgrasp/thirdparty/Grounded-SAM-2` | `git@github.com:IDEA-Research/Grounded-SAM-2.git` |
| `omnidexgrasp/thirdparty/megapose6d` | `git@github.com:megapose6d/megapose6d.git` |

The README will document initialization with:

```bash
git submodule update --init --recursive
```

## Python Environment

Use Python `>=3.10,<3.12`. The existing README targets Python 3.10, and the PyTorch3D/MegaPose/robotics stack is more reliable on Python 3.10 or 3.11 than Python 3.12.

The expected primary setup becomes:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

CUDA-sensitive packages remain explicit install steps where needed:

- PyTorch CUDA wheels
- PyTorch3D source build
- nvdiffrast source build
- editable installs for `CSDF` and `EasyHOI`
- external server repo installs for HaMeR, Grounded-SAM-2, and MegaPose6D

## Dependency Metadata

`pyproject.toml` will include the current `requirements.txt` dependencies plus code-discovered omissions:

- `pandas` for `recons.pose_est`
- `fastapi`, `uvicorn`, `pydantic` for local server wrappers
- `transformers` and `supervision` for the GSAM wrapper
- `google-genai` and `tenacity` for `scripts/gen_human_grasp.py`

If `uv lock` fails because a dependency requires local CUDA compilation, that dependency should move to an optional/manual install section rather than making the base lock unusable.

## README Updates

The README should be revised to:

- Replace conda setup commands with `uv` commands.
- Mention that commands using `python -m recons...`, `python -m optim...`, and `python -m human2robo...` are run from `omnidexgrasp/`.
- Document submodule SSH setup.
- Clarify that checkpoints, datasets, and MANO files are still manual downloads.
- Clarify that `scripts/gen_human_grasp.py` emits `generated_human_grasp_0.png`, while the pipeline expects `generated_human_grasp.png`; users should rename or copy the selected generated image.

## Validation

After implementation, run:

```bash
git submodule status
uv lock
uv sync --no-install-project
python -m compileall -q omnidexgrasp
```

Heavy model entry points should only be smoke-checked up to import/config loading unless the required checkpoints and CUDA runtime are present.
