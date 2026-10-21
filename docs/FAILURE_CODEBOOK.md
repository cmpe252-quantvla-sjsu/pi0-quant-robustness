# E5 failure-mode codebook — DRAFT v0

**Freeze this (v1) before anyone watches an E3/E4 video.** After the freeze, we don't add codes. Anything that doesn't fit goes under `other` with a one-line note.

Each failed episode gets exactly **one primary code**: the first thing that went irrecoverably wrong.

| Code | Label | Use when |
|---|---|---|
| `TNR` | Target not reached | The gripper never gets within grasping range of the correct object; wanders, hovers, or stalls far from it |
| `WRONG_OBJ` | Wrong object | Approaches or grasps an object other than the instruction's target |
| `GRASP_FAIL` | Grasp failed | Reaches the correct object but the grasp closes on nothing or slips immediately |
| `DROP` | Dropped object | Lifts the correct object, then loses it before placement |
| `WRONG_PLACE` | Wrong placement | Carries the correct object but releases it in the wrong location or orientation |
| `TIMEOUT` | Timeout | Making plausible progress but runs out of steps (no earlier code applies) |
| `OTHER` | Other | Collisions, knocking over the scene, simulator glitches; add a note |

## Procedure

1. A script samples up to 30 BF16-success / VE-W4-fail pairs per condition, plus the opposite-direction pairs. It copies the videos to random IDs and writes the key to a file neither coder opens until coding is finished.
2. Both code the **same 30 videos** independently, then compute Cohen's κ and discuss the disagreements.
3. Split the rest. Record `video_id, coder, code, note` in `results/e5_codes.csv`.
4. Unblind and report the distribution of failure modes among the sampled discordant episodes, per condition and setting.

## Edge cases (decide before freezing)

- The robot grasps the wrong object, drops it, then goes for the right one and times out → `WRONG_OBJ` (first irrecoverable error).
- The episode "succeeds" visually but the env says failure → `OTHER` + note. Check the success predicate.
- Lighting makes the target nearly invisible in the video → still code the behavior. Add a note `low-vis`.
