# Setup

Two software environments, one machine. **Environment A** (`vqb_a`) is VLAQuantBench's
original π0 stack and is used for E0. **Environment B** (`vqb_b`) adds LIBERO-Plus and is
used for E1 and every core run. All core episodes run on the same VM in environment B
(proposal Sections 6.3–6.5).

Items marked **TODO** are details Smitha has from the setup session.

## 1. The VM

| Item | Value |
|---|---|
| Name | `qvla-l4` |
| Zone | `us-central1-c` (Google Cloud) |
| GPU | 1 × NVIDIA L4 (23 GB usable) |
| Machine type | `g2-standard-8` |
| OS image | `common-cu129-ubuntu-2204-nvidia-580` (Google Deep Learning VM, Ubuntu 22.04) |
| Boot disk | 150 GB balanced persistent disk (`df` shows 146 GB) |
| Swap | 32 GB added (loading π0 ran out of RAM without it) |
| Shared conda | `/opt/miniforge3` |
| Shared work dir | `/opt/qvla`, group `qvla` |

**Starting and stopping**
- Stop the VM whenever nobody is using it. The disk (and everything on it) is kept.
- If it won't start because the zone is out of L4s, wait about an hour and retry.
- Bunch GPU work into longer sessions. During the core runs, keep it running across
  back-to-back days and stop it only for a gap of a day or more.
- Never use spot/preemptible instances or change the GPU type: all core runs must be on
  the same hardware.

**Once per login: `~/.bashrc`** (same values for both of us)
```bash
umask 002                                   # files you create stay group-writable
export HF_HOME=/opt/qvla/hf_cache           # shared Hugging Face cache and token
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
Also once: `git config --global --add safe.directory /opt/qvla/third_party/VLAQuantBench`
(the clone is owned by Smitha's login).

**`LIBERO_CONFIG_PATH` is set per environment, not in `~/.bashrc`.** Environment A and B
need different LIBERO configs (B points at the LIBERO-Plus tree), so each conda env sets
its own; `conda activate` then switches it automatically:

| Env | `LIBERO_CONFIG_PATH` |
|---|---|
| `vqb_a` | `/opt/qvla/libero_config_a` |
| `vqb_b` | `/opt/qvla/libero_config_b` |

**Migration (next VM session, whoever is on first; log it in `RUN_LOG.md`).** Until Oct 30
both of us had `export LIBERO_CONFIG_PATH=/opt/qvla/libero_config` in `~/.bashrc`.
```bash
cp -r /opt/qvla/libero_config /opt/qvla/libero_config_a      # copy, so nothing breaks midway
conda env config vars set LIBERO_CONFIG_PATH=/opt/qvla/libero_config_a -n vqb_a
# each of us: delete the LIBERO_CONFIG_PATH line from ~/.bashrc, then reconnect
source /opt/miniforge3/bin/activate vqb_a && echo $LIBERO_CONFIG_PATH   # -> .../libero_config_a
```
`conda env config vars set` writes into the env folder, so it needs group write access to
`/opt/miniforge3/envs/vqb_a`. Remove `/opt/qvla/libero_config` once both logins use `_a`.

**Every session**
```bash
source /opt/miniforge3/bin/activate vqb_a   # or vqb_b once it exists
python -c "import mujoco; print(mujoco.__version__)"   # must print 3.3.2 (implementation note 9)
tmux new -s run                       # run anything long inside tmux; the browser SSH drops often
```
Reattach later with `tmux attach -t run`.

## 2. Pinned versions

| Component | Version | Note |
|---|---|---|
| VLAQuantBench | commit `4a2cb7c` | plus our 3 patches (section 4) |
| LeRobot | **0.5.1** | VLAQuantBench pins 0.5.2, which does not exist on PyPI; 0.5.1 matches VLAQuantBench's records (torch 2.10.0) |
| PyTorch | 2.10.0 | |
| Python | 3.12.16 | from the E0a header |
| MuJoCo | **3.3.2** | the setup script installed 3.8.1; re-pinned. ≥3.4.0 changes box–box collision and costs 3–9 Spatial points |
| robosuite | 1.4.0 | |
| LIBERO-Plus | commit `4976dc3` | environment B only |
| Checkpoint | `lerobot/pi0_libero_finetuned_v044` | |

## 3. Environment A (`vqb_a`), built Oct 29

Built from VLAQuantBench's `scripts/env/setup_lerobot.sh`, with these fixes:

1. LeRobot 0.5.2 → 0.5.1 (see above), edited in the clone's `scripts/env/setup_lerobot.sh`.
2. Installed `cmake` and build tools so `egl_probe` compiles:
   `sudo apt-get install -y tmux git libegl1 libgl1 libglib2.0-0 ffmpeg cmake build-essential libegl1-mesa-dev`.
3. Re-pinned `mujoco==3.3.2`.
4. Installed the NVIDIA graphics package for EGL rendering (the image only had the compute
   driver): `sudo apt-get install -y libnvidia-gl-580-server=580.178.04-0ubuntu0.22.04.1`.
5. Added 32 GB swap.
6. Logged in to Hugging Face (needed for the PaliGemma tokenizer). The token is shared on
   the VM; you don't need your own. It was saved owner-only, so other logins got
   `PermissionError`; fixed with `sudo chmod g+r /opt/qvla/hf_cache/token /opt/qvla/hf_cache/stored_tokens`.
   Running `hf auth login` again resets it, so redo the chmod after any new login.

**Check:** `vqb inspect --model pi0` reports the vision encoder as **162 layers / 411.1M
params**, matching VLAQuantBench's published counts.

## 4. Our VLAQuantBench patches

Three small patches in `patches/vlaquantbench/`, applied to the pinned clone:

| Patch | What it does |
|---|---|
| `0001` | Adds the `NOOP` preset: walks the same layers as W4 for the scope and writes each weight back unchanged (implementation note 11) |
| `0002` | Seeds each episode's flow-matching noise from its logged episode seed, so resumed or partial runs keep matched noise |
| `0003` | Default output path is `runs/` instead of `results/`, so a new run can't "resume" a released log |

Apply them once per clone. On the VM the clone is `/opt/qvla/third_party/VLAQuantBench`:
```bash
cd /opt/qvla/third_party/VLAQuantBench
git log -1 --oneline            # must start with 4a2cb7c
git status                      # only scripts/env/setup_lerobot.sh modified (0.5.2 -> 0.5.1)
git switch -c qvla-patches
git am ~/0001-*.patch ~/0002-*.patch ~/0003-*.patch
vqb presets                     # should list NOOP
```
The team repo is not cloned on the VM, so the patch files were uploaded to `~` with the
SSH-in-browser **UPLOAD FILE** button. `vqb_a` already uses this clone in editable mode, so no
reinstall is needed. **Applied Oct 30: `qvla-patches` @ `c497a0d`.**
Environment B starts from a copy of environment A, so it inherits the patches. Record the
patch commit in `RUN_LOG.md`.

## 5. Environment B (`vqb_b`), plan

From implementation note 8 (**verify** while building):
1. Clone `vqb_a` into `vqb_b`.
2. `pip uninstall hf_libero` (both it and LIBERO-Plus install a package named `libero`).
3. `pip install -e LIBERO-plus --no-deps` (its `requirements.txt` pins numpy 1.22.4,
   transformers 4.21.1 and gym 0.25.2, which would break LeRobot).
4. Add `wand scikit-image usd-core` and apt `libmagickwand-dev`.
5. Unzip the 6.4 GB assets into `LIBERO-plus/libero/libero/assets`.
6. Create `/opt/qvla/libero_config_b/config.yaml` pointing at the LIBERO-Plus tree, then
   `conda env config vars set LIBERO_CONFIG_PATH=/opt/qvla/libero_config_b -n vqb_b`
   (the clone of `vqb_a` inherits `_a`; this overrides it).
7. Re-pin `mujoco==3.3.2`.
8. Check: clean BF16 and NOOP parity runs in `vqb_b` (E1).

## 6. On your laptop (analysis only)

The analysis scripts and tests run without a GPU:
1. Clone VLAQuantBench next to this repo, check out `4a2cb7c`, and apply our patches:
   `git am <this repo>/patches/vlaquantbench/*.patch`.
2. Make a virtual environment in that clone and install it: `pip install -e ".[dev]"`
   (plus `scipy` for the McNemar cross-check tests).
3. With that environment active, run from this repo's root:
   ```bash
   python -m pytest -q tests
   python scripts/qrp_bootstrap.py configs/qrp/released_logs_check.yaml --base-dir ../VLAQuantBench
   ```
