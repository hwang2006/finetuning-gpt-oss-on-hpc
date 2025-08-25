# Singularity Build & Package Workflow on HPC

This README summarizes the procedure for building Singularity containers in an HPC environment with **login nodes** (with `--fakeroot`) and **compute nodes** (without `--fakeroot`). It covers using node-local `/tmp` vs shared `/scratch/$USER`.

---

## Filesystems in HPC

- **/tmp (node-local)**
  - Fast, local SSD or RAM-disk.
  - Full POSIX metadata support (no `lutimes` errors).
  - Not shared between nodes.
- **/scratch/$USER (shared)**
  - Large, persistent, visible cluster-wide.
  - Metadata syscalls like `lutimes()` may fail under fakeroot.
  - Best for storing finished images, not for unpacking layers.

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

## Step-by-Step

### 0. Session prep
```bash
mkdir -p /tmp/$USER/singularity/{cache,tmp} /tmp/$USER/sifs /scratch/$USER/sifs
export SINGULARITY_CACHEDIR=/tmp/$USER/singularity/cache
export SINGULARITY_TMPDIR=/tmp/$USER/singularity/tmp
```

### 1. Build sandbox on LOGIN node (/tmp, with fakeroot)
```bash
singularity build --fakeroot --tmpdir "$SINGULARITY_TMPDIR" \
  --sandbox /tmp/$USER/sifs/pt-sandbox pt-2.8.0-cu129-devel.def
```

### 2. Move sandbox to /scratch (shared)
```bash
rsync -a --delete --no-xattrs --no-owner --no-group \
  /tmp/$USER/sifs/pt-sandbox/  /scratch/$USER/sifs/pt-sandbox
```

### 3. Package SIF on a COMPUTE node (no fakeroot)
```bash
# SLURM example
salloc -N 1 -n 1 --mem=16G -p gpu -t 01:00:00

mkdir -p /tmp/$USER/singularity/{cache,tmp}
export SINGULARITY_CACHEDIR=/tmp/$USER/singularity/cache
export SINGULARITY_TMPDIR=/tmp/$USER/singularity/tmp

singularity build --notest --tmpdir "$SINGULARITY_TMPDIR" \
  /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif \
  /scratch/$USER/sifs/pt-sandbox
```

### 4. Validate & run
```bash
singularity exec --nv /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

singularity exec /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif git --version
singularity exec /scratch/$USER/sifs/pt-2.8.0-cu129-devel.sif wget --version
```

---

## Common Pitfalls & Fixes

- **`lutimes ... operation not permitted`**  
  → Build on node-local `/tmp`, not on `/scratch`.

- **`signal: killed`** during packaging  
  → `mksquashfs` ran out of memory. Request more RAM with SLURM (`--mem=16G+`).

- **Sandbox invisible on compute node**  
  → `/tmp` is per-node. Always rsync to `/scratch` for shared access.

- **No fakeroot on compute node**  
  → Packaging doesn’t need fakeroot, only `%post` build does.

---

## Minimal sbatch Template

```bash
#!/bin/bash
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

1. **Build on /tmp (login node, fakeroot)** → run `%post`, install packages.  
2. **Rsync sandbox to /scratch** → shared access.  
3. **Package on compute node (no fakeroot)** → produce SIF.  
4. **Run the SIF anywhere** → portable, immutable container.  
