# UV Submodules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conda-centered setup with a `uv` workflow, add SSH Git submodules for directly referenced external repos, and document a reproducible setup path.

**Architecture:** Keep the local OmniDexGrasp environment managed by root-level `pyproject.toml` and `uv.lock`, with `tool.uv.package = false` because the code is launched from `omnidexgrasp/` using top-level imports such as `recons` and `utils`. Add external model/tool repos as submodules under `omnidexgrasp/thirdparty/`, and keep CUDA/source-build components as explicit install steps in README rather than forcing them into the base sync.

**Tech Stack:** Git submodules, `uv` 0.10.2, Python `>=3.10,<3.12`, PyTorch CUDA 12.8 wheel index, Hydra, Open3D, PyTorch3D/nvdiffrast manual source builds, CSDF/EasyHOI editable installs.

---

### Task 1: Add SSH Submodules

**Files:**
- Create: `.gitmodules`
- Create: `omnidexgrasp/thirdparty/CSDF` gitlink
- Create: `omnidexgrasp/thirdparty/EasyHOI` gitlink
- Create: `omnidexgrasp/thirdparty/hamer` gitlink
- Create: `omnidexgrasp/thirdparty/Grounded-SAM-2` gitlink
- Create: `omnidexgrasp/thirdparty/megapose6d` gitlink

- [ ] **Step 1: Create thirdparty parent directory**

Run:

```bash
mkdir -p omnidexgrasp/thirdparty
```

Expected: command exits 0 and `omnidexgrasp/thirdparty/` exists.

- [ ] **Step 2: Add CSDF submodule**

Run:

```bash
git submodule add git@github.com:wrc042/CSDF.git omnidexgrasp/thirdparty/CSDF
```

Expected: command exits 0 and `.gitmodules` contains a `CSDF` entry.

- [ ] **Step 3: Add EasyHOI submodule**

Run:

```bash
git submodule add git@github.com:lym29/EasyHOI.git omnidexgrasp/thirdparty/EasyHOI
```

Expected: command exits 0 and `.gitmodules` contains an `EasyHOI` entry.

- [ ] **Step 4: Add HaMeR submodule**

Run:

```bash
git submodule add git@github.com:geopavlakos/hamer.git omnidexgrasp/thirdparty/hamer
```

Expected: command exits 0 and `.gitmodules` contains a `hamer` entry.

- [ ] **Step 5: Add Grounded-SAM-2 submodule**

Run:

```bash
git submodule add git@github.com:IDEA-Research/Grounded-SAM-2.git omnidexgrasp/thirdparty/Grounded-SAM-2
```

Expected: command exits 0 and `.gitmodules` contains a `Grounded-SAM-2` entry.

- [ ] **Step 6: Add MegaPose6D submodule**

Run:

```bash
git submodule add git@github.com:megapose6d/megapose6d.git omnidexgrasp/thirdparty/megapose6d
```

Expected: command exits 0 and `.gitmodules` contains a `megapose6d` entry.

- [ ] **Step 7: Verify submodule metadata**

Run:

```bash
git submodule status
sed -n '1,220p' .gitmodules
```

Expected: five submodule status lines are printed. `.gitmodules` uses the exact SSH URLs listed in the design spec.

- [ ] **Step 8: Commit submodules**

Run:

```bash
git add .gitmodules omnidexgrasp/thirdparty
git commit -m "build: add third-party submodules"
```

Expected: commit succeeds and records `.gitmodules` plus five gitlinks.

### Task 2: Add UV Project Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`

- [ ] **Step 1: Create `pyproject.toml`**

Use `apply_patch` to add:

```toml
[project]
name = "omnidexgrasp"
version = "0.1.0"
description = "Generalizable dexterous grasping via foundation models and force feedback"
readme = "README.md"
requires-python = ">=3.10,<3.12"
dependencies = [
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "numpy>=1.24.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "pyyaml>=6.0",
    "manotorch @ git+https://github.com/lixiny/manotorch.git",
    "pytorch-kinematics>=0.7.0",
    "trimesh>=4.0.0",
    "open3d>=0.17.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "pycocotools>=2.0.7",
    "requests>=2.31.0",
    "viser>=0.2.0",
    "scikit-image>=0.20.0",
    "imageio>=2.31.0",
    "pyrender>=0.1.45",
    "mesh-to-sdf>=0.0.14",
    "geomloss>=0.2.0",
    "matplotlib>=3.7.0",
    "rootutils>=1.0.0",
    "pandas>=2.0.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.0.0",
    "transformers>=4.45.0",
    "supervision>=0.22.0",
    "google-genai>=1.0.0",
    "tenacity>=8.2.0",
]

[project.optional-dependencies]
manual-build = [
    "chamfer-distance>=0.1",
]

[tool.uv]
package = false

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Expected: `pyproject.toml` exists and `toml` syntax is valid.

- [ ] **Step 2: Generate `uv.lock`**

Run:

```bash
uv lock
```

Expected: command exits 0 and creates `uv.lock`.

- [ ] **Step 3: Verify base environment sync metadata**

Run:

```bash
uv sync --no-install-project
```

Expected: command exits 0 and creates or updates `.venv` using the locked base dependencies.

- [ ] **Step 4: Commit UV metadata**

Run:

```bash
git add pyproject.toml uv.lock
git commit -m "build: add uv project metadata"
```

Expected: commit succeeds and records both files.

### Task 3: Update README Installation Flow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the conda-oriented installation block**

Replace the current “Setting up the `omnidexgrasp` environment” commands with this `uv` block:

```markdown
**Setting up the `omnidexgrasp` environment with `uv`:**

```bash
# 1. Create and activate the local uv environment
uv venv --python 3.10
source .venv/bin/activate

# 2. Install locked base dependencies
uv sync

# 3. Install PyTorch3D
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# 4. Install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
uv pip install --no-build-isolation /tmp/nvdiffrast

# 5. Install local source-build dependencies from submodules
uv pip install -e omnidexgrasp/thirdparty/CSDF --no-build-isolation
uv pip install -e omnidexgrasp/thirdparty/EasyHOI

# 6. Optional legacy dependency used by older EasyHOI paths
uv pip install "chamfer-distance>=0.1"
```
```

Expected: README no longer tells users to create or activate a conda env for the main OmniDexGrasp environment.

- [ ] **Step 2: Replace submodule setup wording**

Ensure the clone/setup section includes:

```markdown
# 2. Initialize submodules
git submodule update --init --recursive
```

Expected: README no longer implies submodules exist without `.gitmodules`.

- [ ] **Step 3: Replace server environment wording**

Replace the server dependency note with:

```markdown
**Setting up server environments (`hamer`, `gsam`, `megapose`) with `uv`:**

The upstream model stacks still have conflicting dependencies, so keep separate `uv` virtual environments for these modules. Install each upstream project according to its official documentation inside its own environment, then install this repository's lightweight server dependencies:

```bash
# Example: HaMeR server environment
uv venv .venv-hamer --python 3.10
source .venv-hamer/bin/activate
uv pip install -e omnidexgrasp/thirdparty/hamer
uv pip install fastapi uvicorn pydantic hydra-core omegaconf

# Example: Grounded-SAM-2 server environment
uv venv .venv-gsam --python 3.10
source .venv-gsam/bin/activate
uv pip install -e omnidexgrasp/thirdparty/Grounded-SAM-2
uv pip install fastapi uvicorn pydantic hydra-core omegaconf

# Example: MegaPose6D environment
uv venv .venv-megapose --python 3.10
source .venv-megapose/bin/activate
uv pip install -e omnidexgrasp/thirdparty/megapose6d
```
```

Expected: README explains that `uv` replaces conda but does not collapse the conflicting model stacks into one environment.

- [ ] **Step 4: Commit README installation changes**

Run:

```bash
git add README.md
git commit -m "docs: document uv installation"
```

Expected: commit succeeds.

### Task 4: Document Runtime Path And Data Naming Constraints

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clarify working directory for module commands**

In the Usage section, keep or add this sentence before Stage 1 commands:

```markdown
All `python -m ...` commands below are run from the `omnidexgrasp/` directory because the code imports `recons`, `optim`, `human2robo`, and `utils` as top-level packages.
```

Expected: users know not to run `python -m recons.client` from the repository root.

- [ ] **Step 2: Clarify generated grasp image naming**

In the “Generate Grasp Image” subsection, add:

```markdown
If you use `scripts/gen_human_grasp.py`, it writes files named `generated_human_grasp_0.png`, `generated_human_grasp_1.png`, and so on. Pick one generated image and copy or rename it to `generated_human_grasp.png`, which is the filename consumed by `recons.client`, `recons.pose_est`, and `optim.main`.
```

Expected: README explicitly resolves the script/pipeline filename mismatch.

- [ ] **Step 3: Commit README runtime notes**

Run:

```bash
git add README.md
git commit -m "docs: clarify runtime data conventions"
```

Expected: commit succeeds.

### Task 5: Final Verification

**Files:**
- Verify: `.gitmodules`
- Verify: `pyproject.toml`
- Verify: `uv.lock`
- Verify: `README.md`
- Verify: `omnidexgrasp/**/*.py`

- [ ] **Step 1: Verify submodules**

Run:

```bash
git submodule status
```

Expected: five submodule lines are printed for `CSDF`, `EasyHOI`, `hamer`, `Grounded-SAM-2`, and `megapose6d`.

- [ ] **Step 2: Verify lock is current**

Run:

```bash
uv lock --check
```

Expected: command exits 0, confirming `uv.lock` matches `pyproject.toml`.

- [ ] **Step 3: Verify sync works**

Run:

```bash
uv sync --no-install-project
```

Expected: command exits 0 using the lockfile.

- [ ] **Step 4: Verify Python syntax**

Run:

```bash
python -m compileall -q omnidexgrasp
find omnidexgrasp -type d -name __pycache__ -prune -exec rm -rf {} +
```

Expected: `compileall` exits 0, and generated bytecode directories are removed.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: branch is clean after the planned commits, and recent commits include the submodule, uv metadata, and README documentation commits.
