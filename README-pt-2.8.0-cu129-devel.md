# Building `pt-2.8.0-cu129-devel.sif` (PyTorch 2.8.0 • CUDA 12.9 • cuDNN 9 devel)

This guide documents a **reproducible** way to build a Singularity/Apptainer image based on
`ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, add a few essential tools, fix locale, and avoid
common HPC pitfalls (Lustre xattrs, OOM during squashfs, etc.).

> **TL;DR**  
> 1) Create a writable sandbox with `--fakeroot`  
> 2) Inside the sandbox: install `git curl wget locales tzdata` and clean apt caches  
> 3) Copy sandbox to **local disk** (`/tmp/$USER`) to avoid Lustre issues  
> 4) Build SIF on a **compute node** with `/tmp/$USER` as temp/cache dirs  
> 5) Move the SIF back to your repo/scratch

---

## Prerequisites

- Singularity or Apptainer with `--fakeroot` enabled on your cluster (recommended).  
- Access to a **compute node** (via `srun`/`sbatch`) with ≥ 8–16 GB RAM for the build step.  
- Enough free space in `/tmp/$USER` (several GB).

---

## 0) Set variables (optional)
```bash
export REPO_DIR=/scratch/$USER/sifs           # where final .sif will live
export WORK_SANDBOX=$REPO_DIR/pt-2.8.0-cu129-devel
export TMP_STAGE=/tmp/$USER/pt-2.8.0-cu129-devel
mkdir -p "$REPO_DIR" "/tmp/$USER"
```

---

## 1) Create a writable sandbox from the Docker base
```bash
singularity build --sandbox "$WORK_SANDBOX" \
  docker://ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
INFO:    Starting build...
INFO:    Fetching OCI image...
28.2MiB / 28.2MiB [====================================================] 100 % 0.0 b/s 0s
86.9KiB / 86.9KiB [====================================================] 100 % 0.0 b/s 0s
2.1GiB / 2.1GiB [======================================================] 100 % 0.0 b/s 0s
4.4MiB / 4.4MiB [======================================================] 100 % 0.0 b/s 0s
98.7MiB / 98.7MiB [====================================================] 100 % 0.0 b/s 0s
894.8KiB / 894.8KiB [==================================================] 100 % 0.0 b/s 0s
4.4GiB / 4.4GiB [======================================================] 100 % 0.0 b/s 0s
3.0GiB / 3.0GiB [======================================================] 100 % 0.0 b/s 0s
INFO:    Extracting OCI image...
2025/08/24 22:14:31  warn xattr{var/log/apt/term.log} ignoring ENOTSUP on setxattr "user.rootlesscontainers"
2025/08/24 22:14:31  warn xattr{/scratch/qualis/sifs/build-temp-595229858/rootfs/var/log/apt/term.log} destination filesystem does not support xattrs, further warnings will be suppressed
INFO:    Inserting Singularity configuration...
INFO:    Creating sandbox directory...
INFO:    Build complete: /scratch/qualis/sifs/pt-2.8.0-cu129-devel
```

---

## 2) Customize inside the sandbox

Enter the sandbox:
```bash
singularity shell --writable --fakeroot "$WORK_SANDBOX"
WARNING: Skipping mount /scratch [binds]: /scratch doesn't exist in container
WARNING: Skipping mount /home01 [binds]: /home01 doesn't exist in container
WARNING: Skipping mount /apps [binds]: /apps doesn't exist in container
WARNING: Skipping mount /etc/localtime [binds]: /etc/localtime doesn't exist in container
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
Singularity>
```

Inside the container, install extra tools and set locale/timezone:
```bash
set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git curl wget ca-certificates locales tzdata

# Enable & generate en_US.UTF-8 locale (safe default for UTF-8)
sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8 LANGUAGE=en_US:en

# (Optional) set container timezone
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

# Create common bind mount targets (silences warnings on some clusters)
mkdir -p /scratch /home01 /apps

# Clean apt caches and lists to avoid _apt permission issues later
rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*
mkdir -p /var/cache/apt/archives/partial
chmod 755 /var/cache/apt/archives/partial

exit
```

> **Note**: Do **not** set `LC_ALL` permanently. Use `LANG` (and optionally `LANGUAGE`) for persistent locale. See the **Locale deep-dive** section below.

---

## 3) Copy sandbox to local disk to avoid Lustre xattrs
Lustre extended attributes (`lustre.lov`) can make `mksquashfs` noisy/fragile. Copy the sandbox to `/tmp/$USER` and drop xattrs:
```bash
mkdir -p "$TMP_STAGE"
rsync -a --delete --no-xattrs --no-owner --no-group "$WORK_SANDBOX"/ "$TMP_STAGE"/
# If rsync is unavailable:
# cp -a --no-preserve=mode,ownership,xattr "$WORK_SANDBOX"/. "$TMP_STAGE"/
```

---

## 4) Build the final SIF **on a compute node** with local temp/cache dirs
Request a node with sufficient RAM (example: 4 cores, 16 GB RAM):
```bash
srun --pty -p <partition> -c 4 --mem=16G bash
```

Inside that session, keep everything under `/tmp/$USER`:
```bash
mkdir -p /tmp/$USER
export TMPDIR=/tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

# Make squashfs gentler on RAM; flags are ignored if unsupported (safe to leave in)
export SINGULARITY_MKSQUASHFS_ARGS="-no-xattrs -mem 2G"
export SINGULARITY_MKSQUASHFS_PROCS=2
export SINGULARITY_COMPRESSOR=gzip

# Build entirely on local /tmp/$USER, then move the SIF to REPO_DIR
singularity build -F /tmp/$USER/pt-2.8.0-cu129-devel.sif "$TMP_STAGE"/
mv /tmp/$USER/pt-2.8.0-cu129-devel.sif "$REPO_DIR"/
```

**Sanity checks:**
```bash
singularity exec "$REPO_DIR"/pt-2.8.0-cu129-devel.sif git --version
singularity exec "$REPO_DIR"/pt-2.8.0-cu129-devel.sif curl --version
singularity exec "$REPO_DIR"/pt-2.8.0-cu129-devel.sif wget --version
singularity exec "$REPO_DIR"/pt-2.8.0-cu129-devel.sif locale

# Optional GPU check
singularity exec --nv "$REPO_DIR"/pt-2.8.0-cu129-devel.sif python - <<'PY'
import torch, subprocess
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
subprocess.run(["nvidia-smi"])
PY
```

---

## Alternative: batch build script (if interactive compute nodes are busy)

Create `build_sif.sbatch`:
```bash
cat > build_sif.sbatch <<'SB'
#!/bin/bash
#SBATCH -p <partition>
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
set -euo pipefail

mkdir -p /tmp/$USER
export TMPDIR=/tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
export SINGULARITY_MKSQUASHFS_ARGS="-no-xattrs -mem 2G"
export SINGULARITY_MKSQUASHFS_PROCS=2
export SINGULARITY_COMPRESSOR=gzip

singularity build -F /tmp/$USER/pt-2.8.0-cu129-devel.sif "$TMP_STAGE"/
mv /tmp/$USER/pt-2.8.0-cu129-devel.sif "$REPO_DIR"/
SB

sbatch build_sif.sbatch
```

---

## Locale deep‑dive (why warnings happen and how we fixed them)

- **Key variables**:  
  - `LANG` — your default locale (e.g., `en_US.UTF-8`).  
  - `LANGUAGE` — language preference order for messages (`en_US:en`).  
  - `LC_ALL` — **temporary override** of all locale categories; **don’t set permanently**.

- **Why the `setlocale` warning?**  
  If you set `LC_ALL=en_US.UTF-8` but the locale isn’t generated/available inside the container, glibc warns.  
  We fixed this by:
  1) enabling `en_US.UTF-8` in `/etc/locale.gen`  
  2) running `locale-gen en_US.UTF-8`  
  3) persisting `LANG=en_US.UTF-8` and `LANGUAGE=en_US:en` via `update-locale`

- **Check available locales**:
  ```bash
  locale -a | grep -i 'en_US\|c\.utf'
  ```
  You should see `en_US.utf8` and/or `C.utf8`.

- **Do not set `LC_ALL` permanently**:  
  Keep `LC_ALL` **unset** for normal use. Use it only for short-term overrides in a single command.

- **Fallback**: If you don’t need a specific locale, `C.UTF-8` is a minimal, safe default:
  ```bash
  export LANG=C.UTF-8
  ```

---

## Optional: Reproducible build via `.def` (skip the sandbox next time)

Save as `pt-2.8.0-cu129-devel.def`:
```def
Bootstrap: docker
From: ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

%labels
    Maintainer: $USER
    Base: ghcr.io/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
    Note: Adds git/curl/wget + UTF-8 locale + Asia/Seoul tz + mount targets

%environment
    export LANG=en_US.UTF-8
    export LANGUAGE=en_US:en
    export PATH=/opt/conda/bin:$PATH

%post
    set -e
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl wget ca-certificates locales tzdata
    # Locale & timezone
    sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen
    locale-gen en_US.UTF-8
    update-locale LANG=en_US.UTF-8 LANGUAGE=en_US:en
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
    # Bind mount targets commonly used on the cluster
    mkdir -p /scratch /home01 /apps
    # Slim
    apt-get clean
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

%runscript
    exec "$@"
```

Build with local temp/cache dirs (preferably on a compute node):
```bash
mkdir -p /tmp/$USER
export TMPDIR=/tmp/$USER
export SINGULARITY_TMPDIR=/tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
singularity build -F /tmp/$USER/pt-2.8.0-cu129-devel.sif pt-2.8.0-cu129-devel.def
mv /tmp/$USER/pt-2.8.0-cu129-devel.sif "$REPO_DIR"/
```

---

## Cleanup

```bash
rm -rf "$TMP_STAGE"
# (optional) clear the entire temp tree:
# rm -rf /tmp/$USER/*
```

---

## Troubleshooting quick refs

- **`FATAL ... signal: killed` during squashfs** → OOM: build on a compute node with more RAM, set `/tmp/$USER` for temp/cache, reduce `mksquashfs` memory (`-mem 2G`) and procs.
- **`Unrecognised xattr prefix lustre.lov`** → copy sandbox to `/tmp/$USER` with `--no-xattrs` and build from there.
- **`_apt permission denied`** → clean caches and recreate `archives/partial` with 755 perms before repack.
- **Persisting locale without warnings** → do *not* set `LC_ALL` globally; set `LANG` (`en_US.UTF-8` or `C.UTF-8`).

---

**Done!** Your final image lives at:
```
$REPO_DIR/pt-2.8.0-cu129-devel.sif
```
