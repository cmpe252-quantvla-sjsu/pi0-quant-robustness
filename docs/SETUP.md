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
| Machine type, disk size, OS image | **TODO** |
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

**Every session**
```bash
export PYOPENGL_PLATFORM=egl          # also add this line to ~/.bashrc (needed for rendering)
conda activate vqb_a                  # or vqb_b once it exists
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
2. Installed `cmake` and build tools so `egl_probe` compiles. Exact packages: **TODO**.
3. Re-pinned `mujoco==3.3.2`.
4. Installed the NVIDIA graphics package for EGL rendering (the image only had the compute
   driver). Exact package: **TODO**.
5. Added 32 GB swap.
6. Logged in to Hugging Face (needed for the PaliGemma tokenizer). The token is shared on
   the VM; you don't need your own.

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
git am <team repo on the VM>/patches/vlaquantbench/*.patch     # path: TODO
pip install -e .                # inside the conda env; reinstalls the patched code
vqb presets | grep NOOP         # should list NOOP
```
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
6. Point `~/.libero/config.yaml` at the LIBERO-Plus tree.
7. Re-pin `mujoco==3.3.2`.
8. Check: clean BF16 and NOOP parity runs in `vqb_b` (E1).

## 6. On your laptop (analysis only)

The analysis scripts and tests run without a GPU:
```powershell
cd C:\VLAQuantBench\VLAQuantBench        # VLAQuantBench clone with the patches applied
.venv\Scripts\Activate.ps1
cd ..\pi0-quant-robustness
python -m pytest -q tests
```
