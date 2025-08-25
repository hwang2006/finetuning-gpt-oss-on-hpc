# HPC Container Build Guide

This document explains how to build and package custom Singularity (SIF) containers for HPC environments.  
It merges both the **practical workflow** and the **background rationale** so you have everything in one place.

---

## Why build your own container?

- **Missing tools in base images**  
  Base PyTorch images (from DockerHub or GHCR) are often slim. They may lack:
  - `git` (needed because many Python packages install directly from GitHub via pip)
  - `wget` / `curl` (needed for fetching data/models)
  - Proper `locales` (to avoid `setlocale` warnings in Python)
  - `tzdata` (timezone settings)
- **Creating Python virtual environments**  
  To use pip reliably, `git` and HTTPS tools are required inside the container.  
  Without them, you cannot build environments that depend on GitHub-hosted packages.
- **Cluster portability**  
  Adding bind mount points like `/scratch`, `/apps`, `/home01` ensures paths exist when the container runs on HPC nodes.
- **Locale & Timezone**  
  Ensuring `en_US.UTF-8` and correct timezone avoids annoying warnings and improves reproducibility.

---

## Filesystems in HPC

- **/tmp (node-local)**
  - Fast local SSD or RAM-disk.
  - Full POSIX metadata support (avoids `lutimes` fakeroot errors).
  - Ephemeral (deleted when job ends).
  - Not shared between nodes.

- **/scratch/$USER (shared)**
  - Large, persistent, visible cluster-wide.
  - May cause metadata syscalls like `lutimes()` to fail under fakeroot.
  - Best for storing **finished SIF images**, datasets, checkpoints.

---

## Lifecycle Diagram

```
┌──────────────────────────────┐
│ 0) Definition (.def) file    │
│    - Bootstrap: docker       │
│    - %post: install pkgs     │
│    - %env, %runscript        │
└───────────────┬──────────────┘
                │ (pull base + run %post)
                ▼
┌──────────────────────────────────────────┐
│ 1) Build SANDBOX on LOGIN node (/tmp)    │
│    - Needs --fakeroot                    │
│    - Writable rootfs                     │
│    - Avoids lutimes errors               │
└───────────────┬──────────────────────────┘
                │ (make visible cluster-wide)
                ▼
┌──────────────────────────────────────────┐
│ 2) RSYNC sandbox → /scratch/$USER        │
│    - Shared FS, accessible by all nodes  │
└───────────────┬──────────────────────────┘
                │ (compress with RAM/IO)
                ▼
┌──────────────────────────────────────────┐
│ 3) PACKAGE SIF on COMPUTE node           │
│    - No fakeroot required                │
│    - Uses mksquashfs (needs RAM)         │
│    - Use compute-node /tmp as TMPDIR     │
│    - Output to /scratch/$USER            │
└───────────────┬──────────────────────────┘
                │ (immutable, portable)
                ▼
┌──────────────────────────────────────────┐
│ 4) RUN the SIF anywhere                  │
│    - singularity exec --nv ...           │
│    - Bind /scratch, /home01, /apps       │
└──────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### 0. Session prep
Always redirect Singularity cache/tmp to node-local `/tmp` to avoid metadata errors.

```bash
mkdir -p /tmp/$USER/singularity/{cache,tmp} /tmp/$USER/sifs /scratch/$USER/sifs
export SINGULARITY_CACHEDIR=/tmp/$USER/singularity/cache
export SINGULARITY_TMPDIR=/tmp/$USER/singularity/tmp
export REPO=/scratch/$USER/finetuning-gpt-oss-on-hpc
```

### 1. Build sandbox on LOGIN node (/tmp, with fakeroot)

Here you run `%post`, install missing tools (`git`, `wget`, `curl`, `locales`, etc.), configure locale/timezone.

```bash
singularity build --fakeroot --tmpdir "$SINGULARITY_TMPDIR"  --sandbox /tmp/$USER/sifs/pt-sandbox "$REPO/pt-2.8.0-cu129-devel.def"
```

### 2. Move sandbox to /scratch (shared)

Since `/tmp` is node-local, rsync to `/scratch/$USER` for cluster-wide visibility.

```bash
rsync -a --delete --no-xattrs --no-owner --no-group   /tmp/$USER/sifs/pt-sandbox/  /scratch/$USER/sifs/pt-sandbox
```

### 3. Package SIF on a COMPUTE node (no fakeroot)

Packaging compresses the sandbox into a `.sif` image. Needs RAM for `mksquashfs`.

```bash
# Example: request interactive GPU/CPU job
salloc -N 1 -n 1 --mem=16G -p amd_a100nv_8 -t 02:00:00 --comment pytorch

mkdir -p /tmp/$USER/singularity/{cache,tmp}
export SINGULARITY_CACHEDIR=/tmp/$USER/singularity/cache
export SINGULARITY_TMPDIR=/tmp/$USER/singularity/tmp

singularity build --notest --tmpdir "$SINGULARITY_TMPDIR"   /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif   /scratch/$USER/sifs/pt-sandbox
```

### 4. Validate & run

```bash
singularity exec --nv /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

singularity exec /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif git --version
singularity exec /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif wget --version
```

---

## Common Pitfalls & Fixes

- **`lutimes ... operation not permitted`**  
  → Happens when fakeroot builds on `/scratch`. Fix: build sandbox in `/tmp`.

- **`signal: killed` during packaging**  
  → `mksquashfs` ran out of memory. Request more RAM (`--mem=16G+`).

- **Sandbox invisible on compute node**  
  → `/tmp` is per-node. Must rsync to `/scratch`.

- **No fakeroot on compute node**  
  → That’s fine. Packaging (`mksquashfs`) doesn’t need fakeroot, only `%post` does.

---

## Minimal sbatch Template

```bash
#!/bin/bash
#SBATCH --comment pytorch
#SBATCH -J torch28
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -p gpu

export SINGULARITY_CACHEDIR=/tmp/$USER/singularity/cache
export SINGULARITY_TMPDIR=/tmp/$USER/singularity/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

SIF=/scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif

singularity exec --nv -B /scratch:/scratch "$SIF" bash -lc '
python - <<PY
import torch, time
print("now:", time.strftime("%F %T %Z"))
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
PY'
```

---

## Key Takeaways

1. **Build sandbox on /tmp (login node with fakeroot)** → run `%post`, install tools.  
2. **Rsync sandbox to /scratch** → make it visible cluster-wide.  
3. **Package SIF on compute node (no fakeroot)** → `mksquashfs` needs RAM.  
4. **Run final `.sif` anywhere** → portable, immutable container with all dependencies.
