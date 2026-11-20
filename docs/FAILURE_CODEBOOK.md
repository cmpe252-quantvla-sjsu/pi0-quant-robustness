# E5 failure codebook

Labels for the blinded failure-mode coding in E5 (proposal §5, "Failure-mode coding"). Every
sampled video shows an episode that **failed**; each coder gives it exactly **one code** from
the table below.

**Status: v1, frozen when this file is merged into `main`.** It was drafted from E0a pilot
videos only (20 BF16 failures on clean LIBERO-Spatial, listed at the end). No E2 or E3 core
video was watched before the freeze. After the freeze the codes and rules do not change; a
necessary change is logged in `RUN_LOG.md` and reported as a deviation.

## Codes

A task has up to four stages: **reach** the target object → **grasp** it → **carry** it →
**place** it (or finish the final action). Most codes name the stage where the episode broke
down. "Target object" is the object the instruction names (e.g. *the black bowl on the
stove*); "goal" is where it should end up, or the final state the instruction asks for (e.g.
the drawer open).

| Code | Name | Definition | Pilot examples (E0a) |
|---|---|---|---|
| `NO_REACH` | Target not reached | The gripper never gets to the target object: it freezes, moves aimlessly, or never comes close enough to try a grasp. Touching the container the target is in or on (drawer, cabinet, box) while trying for it counts as reaching it (rule 4). | none |
| `WRONG_OBJECT` | Wrong object | The robot **aims at a different object** than the target, somewhere else in the scene (e.g. the other bowl), and goes for, grasps or moves it as if it were the target. Not for an object stacked under or attached to the target (that is `EXTRA_OBJECT`, rule 5). | none |
| `GRASP_MISS` | Grasp failed | The gripper gets to the target and closes (on it, on its edge, or on air right next to it), but the target is **never lifted** (rule 2). Pushing, nudging or tipping it does not count as lifting. | t0_e2, t1_e9, t3_e1, t4_e1, t4_e5, t6_e3 |
| `SLIP` | Dropped object | The target is **lifted** (rule 2), then falls or slides out of the gripper **before** the robot releases it at the goal. | t1_e6, t2_e5, t2_e10, t3_e4, t8_e0 |
| `EXTRA_OBJECT` | Extra object moved | The robot aims at the target, but another object is moved with it or instead of it: an object stacked under or attached to the target (e.g. the ramekin under the bowl), or one caught by the gripper together with the target (e.g. the plate's edge). The other object spoils the result. | t5_e1, t8_e1 |
| `PLACE_MISS` | Wrong placement | The target reaches the goal area and is **released**, but the result does not count: tilted on the edge, off-center or next to the goal, or the final action is done on the right object but not completed (e.g. drawer only partly open). | t5_e3, t6_e2, t7_e4, t7_e13, t9_e0, t9_e1 |
| `TIMEOUT` | Timeout | At the furthest stage reached, the robot's last attempt shows **no visible mistake**: it is still holding the target (e.g. over the goal) or carrying it slowly when time runs out. | t0_e1 |
| `OTHER` | Other | None of the above fits. Write what happened in the note. | none |

## Decision rules

1. **One code per video.** If the robot tries more than once, find the **furthest stage it
   reached** (reach → grasp → carry → place) and code how its **last attempt at that stage**
   ended. Examples: lifted, slipped, grasped again, carried to the plate, still holding it
   when time runs out → `TIMEOUT`; lifted and slipped over the plate twice → `SLIP`; lifted,
   slipped, re-grasped, released tilted on the plate → `PLACE_MISS`.
2. **Lifted** means the target is **held between the gripper's fingers with its bottom off
   the surface**. A target that moves only because the arm or the outside of the fingers
   pushed, nudged, slid or tipped it was **not** lifted. Lifted or not separates `GRASP_MISS`
   from `SLIP`; if you cannot tell, use `GRASP_MISS` and say so in the note.
3. **Wandering after a failure is not `TIMEOUT`.** Almost every failed episode ends with the
   arm hovering until time runs out; code the mistake that came before it. Use `TIMEOUT` only
   when there is no earlier mistake at the furthest stage.
4. **Containers and incidental contact.** Touching or grabbing the drawer, cabinet or box that
   holds the target, while trying for the target, counts as reaching the target: if the
   target is then never lifted, it is `GRASP_MISS`, not `NO_REACH` or `WRONG_OBJECT`. Bumping
   other objects along the way is never `WRONG_OBJECT` on its own.
5. **`EXTRA_OBJECT` vs `WRONG_OBJECT`** depends on **what the robot aims at**. Aiming at the
   target but moving something stacked under it, attached to it, or caught with it →
   `EXTRA_OBJECT`, even if the gripper ends up holding only the other object. Going to a
   different object elsewhere in the scene → `WRONG_OBJECT`.
6. **Released at the goal or not** separates `SLIP` from `PLACE_MISS`: a target that falls
   while still being carried, before the gripper opens over the goal, is `SLIP`.
7. **Unsure between two codes:** pick the one at the **earlier** stage and explain in the note.
   Use `OTHER` only when no code fits, not when two do.

## How to code

1. Open `coding_sheet.csv` from the E5 folder. Code only rows where `assigned` is your letter
   or `both`. Coder A is Amina, coder B is Smitha.
2. Videos are named by a random ID (`videos/<video_id>.mp4`); the `instruction` column says
   which object is the target. Watch as many times as needed, at reduced speed if useful.
3. Enter the code (exactly as in the table, e.g. `PLACE_MISS`) in `code_A` or `code_B`, and a
   short note in `note_A` or `note_B` (one line: what happened, and the rule used if it was a
   close call).
4. **Never open the `key/` folder** before all coding is finished. It says which setting
   failed, which would unblind the coding.
5. Code the 30 shared (`both`) videos **independently**: do not discuss them or look at the
   other coder's column until both are done. Then run `kappa`, and resolve each disagreement
   together in `final_code`. κ is reported on the independent codes, before resolution.
6. `final_code` is only needed for shared videos where the two codes differ. Otherwise
   `report` uses the agreed code (shared videos) or the single coder's code.

## Test before the freeze

On 2026-11-20 the draft was tested on 10 E0a failures not used to write it (t0_e5, t1_e11,
t3_e20, t4_e6, t5_e6, t6_e19, t7_e15, t8_e2, t9_e5, t9_e6), coded twice independently (Amina
watching the videos; an AI-assisted second pass from still frames). The codes agreed on 6 of 10. The four
disagreements came from unclear wording, and led to the current rule 1 (last attempt at the
furthest stage), rule 2 (pushed is not lifted), rule 4 (containers) and rule 5 (stacked
objects are `EXTRA_OBJECT`). With these rules both coders agree on all 10. This test is not
the E5 κ, which is computed on the 30 shared core videos.

## Sampling (pre-registered)

Decided 2026-11-13 (`RUN_LOG.md`, Decisions). The video sampled for a pair is the episode that
**failed**.

- **All VE-W4 discordant pairs** (VE-W4 vs BF16 on the same episode, one succeeds, the other
  fails) in every core condition: clean (E2) and lighting, background, layout (E3). 56 pairs.
- **W4 discordant pairs: up to 15 per condition and direction** (BF16 fails / W4 fails), drawn
  with seed 0, so the failure modes of a setting with a larger quantization effect can be
  compared with VE-W4.
- The same BF16 failure video is used once even if it belongs to a VE-W4 and a W4 pair.
  Expected total: 166 videos, 30 of them shared.
- Command (on the VM, after this codebook is merged):
  `python scripts/e5_videos.py sample --videos /opt/qvla/videos --out /opt/qvla/e5 --w4-per-direction 15 --seed 0`
  The sample is drawn once; it is never redrawn after coding starts.

## Reporting

`report` gives the code counts per condition, failing setting and direction. We report them as
failure modes among the **sampled discordant episodes** (descriptive, no test). More
`WRONG_OBJECT` or `NO_REACH` among quantized failures than among BF16 failures would be
consistent with perception-related failures (proposal §5); grasp, carry and placement codes
point to action precision instead.

## Relation to the proposal

The proposal lists target not reached, wrong object, grasp failed, dropped object, wrong
placement, timeout and other; these are `NO_REACH`, `WRONG_OBJECT`, `GRASP_MISS`, `SLIP`,
`PLACE_MISS`, `TIMEOUT` and `OTHER`. `EXTRA_OBJECT` is added because it appeared in 2 of the
20 pilot failures and fits none of the listed codes.

## Pilot videos used for drafting

E0a, BF16, clean LIBERO-Spatial (`/opt/qvla/videos/E0a/`), two failures per task:
t0_e1, t0_e2, t1_e6, t1_e9, t2_e5, t2_e10, t3_e1, t3_e4, t4_e1, t4_e5, t5_e1, t5_e3, t6_e2,
t6_e3, t7_e4, t7_e13, t8_e0, t8_e1, t9_e0, t9_e1 (files `libero_spatial_<task>_<episode>_fail.mp4`).
These are clean scenes from one suite and one setting; failure types that only appear under
shift or quantization are covered by `NO_REACH`, `WRONG_OBJECT` and `OTHER`.
