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
Singularity> set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git curl wget ca-certificates locales tzdata
Get:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease [1581 B]
Get:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  Packages [1939 kB]
Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]
Get:5 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [48.5 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
Get:7 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1271 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]
Get:11 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [3253 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB]
Get:14 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [5235 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [75.9 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1575 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [5430 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [3569 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [35.2 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [83.2 kB]
Fetched 42.9 MB in 5s (8609 kB/s)
Reading package lists... Done
W: Download is performed unsandboxed as root as file '/var/lib/apt/lists/partial/archive.ubuntu.com_ubuntu_dists_jammy_InRelease' couldn't be accessed by user '_apt'. - pkgAcquire::Run (13: Permission denied)
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
ca-certificates is already the newest version (20240203~22.04.1).
The following additional packages will be installed:
  git-man libbrotli1 libcurl3-gnutls libcurl4 liberror-perl libexpat1 libnghttp2-14
  libpsl5 librtmp1 libssh-4
Suggested packages:
  gettext-base git-daemon-run | git-daemon-sysvinit git-doc git-email git-gui gitk
  gitweb git-cvs git-mediawiki git-svn
Recommended packages:
  less ssh-client publicsuffix
The following NEW packages will be installed:
  curl git git-man libbrotli1 libcurl3-gnutls libcurl4 liberror-perl libexpat1
  libnghttp2-14 libpsl5 librtmp1 libssh-4 locales tzdata wget
0 upgraded, 15 newly installed, 0 to remove and 27 not upgraded.
Need to get 10.6 MB of archives.
After this operation, 47.7 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libexpat1 amd64 2.4.7-1ubuntu0.6 [92.1 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 locales all 2.35-0ubuntu3.10 [4248 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 tzdata all 2025b-0ubuntu0.22.04.1 [347 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libnghttp2-14 amd64 1.43.0-1ubuntu0.2 [76.9 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 libpsl5 amd64 0.21.0-1.2build2 [58.4 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 wget amd64 1.21.2-2ubuntu1.1 [339 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy/main amd64 libbrotli1 amd64 1.0.9-2build6 [315 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 librtmp1 amd64 2.4+20151223.gitfa8646d.1-2build4 [58.2 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libssh-4 amd64 0.9.6-2ubuntu0.22.04.4 [187 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libcurl4 amd64 7.81.0-1ubuntu1.20 [289 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 curl amd64 7.81.0-1ubuntu1.20 [194 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libcurl3-gnutls amd64 7.81.0-1ubuntu1.20 [284 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/main amd64 liberror-perl all 0.17029-1 [26.5 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 git-man all 1:2.34.1-1ubuntu1.15 [955 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 git amd64 1:2.34.1-1ubuntu1.15 [3166 kB]
Fetched 10.6 MB in 4s (2653 kB/s)
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
        LANGUAGE = (unset),
        LC_ALL = "en_US.UTF-8",
        LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
debconf: delaying package configuration, since apt-utils is not installed
Selecting previously unselected package libexpat1:amd64.
(Reading database ... 15182 files and directories currently installed.)
Preparing to unpack .../00-libexpat1_2.4.7-1ubuntu0.6_amd64.deb ...
Unpacking libexpat1:amd64 (2.4.7-1ubuntu0.6) ...
Selecting previously unselected package locales.
Preparing to unpack .../01-locales_2.35-0ubuntu3.10_all.deb ...
Unpacking locales (2.35-0ubuntu3.10) ...
Selecting previously unselected package tzdata.
Preparing to unpack .../02-tzdata_2025b-0ubuntu0.22.04.1_all.deb ...
Unpacking tzdata (2025b-0ubuntu0.22.04.1) ...
Selecting previously unselected package libnghttp2-14:amd64.
Preparing to unpack .../03-libnghttp2-14_1.43.0-1ubuntu0.2_amd64.deb ...
Unpacking libnghttp2-14:amd64 (1.43.0-1ubuntu0.2) ...
Selecting previously unselected package libpsl5:amd64.
Preparing to unpack .../04-libpsl5_0.21.0-1.2build2_amd64.deb ...
Unpacking libpsl5:amd64 (0.21.0-1.2build2) ...
Selecting previously unselected package wget.
Preparing to unpack .../05-wget_1.21.2-2ubuntu1.1_amd64.deb ...
Unpacking wget (1.21.2-2ubuntu1.1) ...
Selecting previously unselected package libbrotli1:amd64.
Preparing to unpack .../06-libbrotli1_1.0.9-2build6_amd64.deb ...
Unpacking libbrotli1:amd64 (1.0.9-2build6) ...
Selecting previously unselected package librtmp1:amd64.
Preparing to unpack .../07-librtmp1_2.4+20151223.gitfa8646d.1-2build4_amd64.deb ...
Unpacking librtmp1:amd64 (2.4+20151223.gitfa8646d.1-2build4) ...
Selecting previously unselected package libssh-4:amd64.
Preparing to unpack .../08-libssh-4_0.9.6-2ubuntu0.22.04.4_amd64.deb ...
Unpacking libssh-4:amd64 (0.9.6-2ubuntu0.22.04.4) ...
Selecting previously unselected package libcurl4:amd64.
Preparing to unpack .../09-libcurl4_7.81.0-1ubuntu1.20_amd64.deb ...
Unpacking libcurl4:amd64 (7.81.0-1ubuntu1.20) ...
Selecting previously unselected package curl.
Preparing to unpack .../10-curl_7.81.0-1ubuntu1.20_amd64.deb ...
Unpacking curl (7.81.0-1ubuntu1.20) ...
Selecting previously unselected package libcurl3-gnutls:amd64.
Preparing to unpack .../11-libcurl3-gnutls_7.81.0-1ubuntu1.20_amd64.deb ...
Unpacking libcurl3-gnutls:amd64 (7.81.0-1ubuntu1.20) ...
Selecting previously unselected package liberror-perl.
Preparing to unpack .../12-liberror-perl_0.17029-1_all.deb ...
Unpacking liberror-perl (0.17029-1) ...
Selecting previously unselected package git-man.
Preparing to unpack .../13-git-man_1%3a2.34.1-1ubuntu1.15_all.deb ...
Unpacking git-man (1:2.34.1-1ubuntu1.15) ...
Selecting previously unselected package git.
Preparing to unpack .../14-git_1%3a2.34.1-1ubuntu1.15_amd64.deb ...
Unpacking git (1:2.34.1-1ubuntu1.15) ...
Setting up libexpat1:amd64 (2.4.7-1ubuntu0.6) ...
Setting up libpsl5:amd64 (0.21.0-1.2build2) ...
Setting up wget (1.21.2-2ubuntu1.1) ...
Setting up libbrotli1:amd64 (1.0.9-2build6) ...
Setting up libnghttp2-14:amd64 (1.43.0-1ubuntu0.2) ...
Setting up locales (2.35-0ubuntu3.10) ...
Generating locales (this might take a while)...
Generation complete.
Setting up tzdata (2025b-0ubuntu0.22.04.1) ...

Current default time zone: 'Etc/UTC'
Local time is now:      Sun Aug 24 13:24:23 UTC 2025.
Universal Time is now:  Sun Aug 24 13:24:23 UTC 2025.
Run 'dpkg-reconfigure tzdata' if you wish to change it.

Setting up liberror-perl (0.17029-1) ...
Setting up librtmp1:amd64 (2.4+20151223.gitfa8646d.1-2build4) ...
Setting up libssh-4:amd64 (0.9.6-2ubuntu0.22.04.4) ...
Setting up libcurl4:amd64 (7.81.0-1ubuntu1.20) ...
Setting up git-man (1:2.34.1-1ubuntu1.15) ...
Setting up curl (7.81.0-1ubuntu1.20) ...
Setting up libcurl3-gnutls:amd64 (7.81.0-1ubuntu1.20) ...
Setting up git (1:2.34.1-1ubuntu1.15) ...
Processing triggers for libc-bin (2.35-0ubuntu3.10) ...
W: Download is performed unsandboxed as root as file '/var/cache/apt/archives/partial/libexpat1_2.4.7-1ubuntu0.6_amd64.deb' couldn't be accessed by user '_apt'. - pkgAcquire::Run (13: Permission denied)

# Enable & generate en_US.UTF-8 locale (safe default for UTF-8)
Singularity> sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen
Singularity> locale-gen en_US.UTF-8
Singularity> update-locale LANG=en_US.UTF-8 LANGUAGE=en_US:en
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
        LANGUAGE = (unset),
        LC_ALL = "en_US.UTF-8",
        LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
*** update-locale: Error: invalid locale settings:  LANG=en_US.UTF-8 LANGUAGE=en_US:en
Singularity> unset LC_ALL
Singularity> unset LANGUAGE
Singularity> export LANG=C.UTF-8
Singularity> rm -f /etc/profile.d/locale.sh
Singularity> sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
Singularity> locale-gen en_US.UTF-8
Generating locales (this might take a while)...
  en_US.UTF-8... done
Generation complete.
Singularity> cat > /etc/default/locale <<'EOF'
LANG=en_US.UTF-8
LANGUAGE=en_US:en
EOF
Singularity> LC_ALL=C locale -a | grep -i 'en_US\|c\.utf'
C.utf8
en_US.utf8
Singularity> exit
exit

# Re-enter and confirm
singularity shell --writable --fakeroot /scratch/$USER/sifs/pt-2.8.0-cu129-devel
WARNING: Skipping mount /scratch [binds]: /scratch doesn't exist in container
WARNING: Skipping mount /home01 [binds]: /home01 doesn't exist in container
WARNING: Skipping mount /apps [binds]: /apps doesn't exist in container
Singularity> locale
LANG=en_US.UTF-8
LANGUAGE=
LC_CTYPE="en_US.UTF-8"
LC_NUMERIC="en_US.UTF-8"
LC_TIME="en_US.UTF-8"
LC_COLLATE="en_US.UTF-8"
LC_MONETARY="en_US.UTF-8"
LC_MESSAGES="en_US.UTF-8"
LC_PAPER="en_US.UTF-8"
LC_NAME="en_US.UTF-8"
LC_ADDRESS="en_US.UTF-8"
LC_TELEPHONE="en_US.UTF-8"
LC_MEASUREMENT="en_US.UTF-8"
LC_IDENTIFICATION="en_US.UTF-8"
LC_ALL=en_US.UTF-8

# (Optional) set container timezone
Singularity> ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

# Create common bind mount targets (silences warnings on some clusters)
Singularity> mkdir -p /scratch /home01 /apps

# Clean apt caches and lists to avoid _apt permission issues later
Singularity> rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*
Singularity> mkdir -p /var/cache/apt/archives/partial
Singularity> chmod 755 /var/cache/apt/archives/partial

Singularity>exit
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
