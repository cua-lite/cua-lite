# All-48 reference audit and implementation plan

This is the preserved 2026-09-11 audit and baseline evaluation history. For the
subsequent 21 local mechanic variants and remaining backend/media gaps, see
[LOCAL_EXPANSION.md](/examples/not_a_robot/LOCAL_EXPANSION.md). Historical
completion counts and model scores below do not grade those new variants.

Audit date: 2026-09-11 UTC. Scope: all 48 synthetic Neal game tasks, including
independent task entry and eventual sequential campaign evaluation. This report
tracks evidence and implementation progress, not completion of all 48 tasks.
The existing first-ten code and prior local fixes are preserved.
The later incremental evidence update and version 0.6.0 refinement are described
in [INCREMENTAL.md](/examples/not_a_robot/INCREMENTAL.md). They add two reference
instances and refine level six, not new implemented levels or replacement scores.

## Verdict

The supplied material is sufficient to start substantial implementation work
across the full catalog. It is **not sufficient to faithfully finish and verify
all 48 tasks**. Missing evidence for individual tasks should not prevent work on
the well-specified tasks. Conversely, a UI shell, guessed answer, recorded-dialogue
replay or simplified opponent must not be counted as a completed reconstruction.

Current code implements 25 captured-instance task paths: levels 01–14, 18, 21–22,
24, 29–31, 33–34, 39 and 46, not all 48. Level 39 is specifically its observed
unavailable-camera branch, not a facial-expression detector. The first expansion
added 12 interactive tasks; the next adds replacements, moving ducks and this
limited camera path, not placeholder cards.
See [current support](/examples/not_a_robot/SUPPORT.md) for
private asset imports and launch commands. The earlier eight authored exercises
are separate. Nine distinct first-ten tasks have local Astra success trajectories
across development attempts, and a bounded expansion smoke succeeded on 21 and
34. A subsequent bounded smoke succeeded on 18, 22 and only the unavailable-camera
path of 39. A later fresh attempt completed all four stages of 24; level 11's
bounded attempt did not solve the task. An interrupted earlier 24 attempt is
preserved but excluded from evaluation. Those development attempts are historical,
not a single-run accuracy score.

A subsequent **fresh run tested all 25 implemented paths** with one independent
Codex attempt each (`gpt-6-astra` / `xhigh`, seed 0, 600 seconds and 150 environment
steps): **22 succeeded; 08, 09 and 11 stopped without success**. All 25 trajectories
passed event/image, model-input provenance, dispatch-pairing and cleanup checks.
The previous development success on 09 does not replace its fresh failure.
This is fixed-instance local coverage, not original-site parity or an all-48
campaign result. No game rules changed during testing.
See [validation](/examples/not_a_robot/VALIDATION.md).

An ordered [local campaign controller](/examples/not_a_robot/CAMPAIGN.md) now
preserves browser/trajectory ownership across task boundaries. It starts at 01,
does not skip missing implementations and cannot yet complete all 48. The static
gallery remains independent-task preview, not a second campaign controller.
A real GUI-oracle regression now completes 01–14 in one campaign and confirms
the actual missing-15 stop with reward 0. It uses test answers and the existing
local opponent, not a vision-model policy or proof of original-site fidelity.

## Input identity and verification

Inspected input:
`neal_levels31_48_reference_20260910_040601.zip`.

- Size: 383,073,153 bytes.
- SHA256: `65af97b1434dfdc60ed8bae99b48941edcbdc14d5db1e24f0456d5e0f7769c40`.
- Byte-identical to the earlier attachment with the same basename.
- Despite the basename, it contains levels 11–48, older duplicate 11–30 trees,
  diagnostics, per-level ZIPs and work files. It contains no task directories
  for levels 1–10.
- The 20 newer 11–30 per-level ZIPs and 18 31–48 ZIPs passed independent package
  size/hash verification. Their manifests' **2,026 file records** passed exact
  file-set, size and SHA256 checks: 89,158,244 bytes hashed, no mismatches.
  These are manifest records, including repeated common files, not 2,026 unique
  gameplay assets. This was not a full decode or security audit of all outer
  archive work files.
- No bundled scripts, chess-engine executable or other programs were executed.
  No original game was accessed or modified during this audit.

Canonical trees inside the ZIP:

```text
P11 = neal_levels11_30_complete_20260910_025702/
P31 = neal_levels31_48_reference_20260910_040601/
```

Use P11 rather than double-counting its older source tree. For P31, prefer
`tasks/level_NN/reviewed_rules.md` and reviewed observations over historical
`rules.md`; the latter contains explanations that were explicitly corrected.

## Evidence coverage is not implementation coverage

| Reference set | Task coverage | Saved official completion records, by distinct level | Other terminal evidence |
| --- | --- | --- | --- |
| Earlier first-ten supplement | 1–10 | 9: 1–5, 7–10 | Level 6 has no winning completion |
| P11 | 11–30 | 12: 11, 12, 13, 14, 18, 20, 21, 22, 24, 28, 29, 30 | Eight levels remain unconfirmed |
| P31 | 31–48 | 8: 31, 33, 34, 36, 39, 42, 46, 47 | Level 48 has a same-attempt visible terminal screen, but no completion message |
| Incremental 20260910_01 | Attempts on 01, 06, 07, 08, 41 | 0 | Visible successes on 01, 06, 07, 08; 41 remains partial |

Combined: **29 levels have saved official completion messages**; levels 06 and
48 have separate visible terminal evidence. The other 17 have neither:
`15, 16, 17, 19, 23, 25, 26, 27, 32, 35, 37, 38, 40, 41, 43, 44, 45`.
Even an observed successful input does not recover a general scoring function.

P11 has 42 attempts: 12 success, 17 unknown, 12 blocked and one rejected. P31
has 25 attempts: eight success, eight unknown, three blocked and six rejected.
The 20 saved completion messages from these two trees match their same-attempt
registration identities, official origin, source/type validation fields and
outcome event references. This establishes internal consistency of the saved
records, not an independent replay of the original site.

The audited new trees have **zero continuous gameplay recordings**. P31 also
lacks saved synchronized audio; the level 47/48 media URLs are not media files.
Many captures use DOM/AX assistance; level 44 uses an external chess engine.
Some observations in levels 19/35/40 disclose hidden text or objects through tool
errors. Preserve these provenance flags; these are implementation references,
not clean screenshot-only model benchmark trajectories.

## Per-level implementation readiness

“Start” means a captured-instance implementation can proceed after matching
assets to its successful screenshots. It does not certify unobserved variants,
exact timing, general decision boundaries or redistribution rights.
“Partial” means useful components can be built, but the listed missing mechanism
prevents faithful end-to-end acceptance. “Existing” refers only to local code.

| Level | Mechanism | Reference outcome | Current readiness / remaining evidence |
| --- | --- | --- | --- |
| 01 | Checkbox | Message | Existing; loading timing and click boundaries inferred |
| 02 | Stop-sign grid | Message | Existing captured instance; unseen photos/boundaries unknown |
| 03 | Distorted moving text | Message | Existing; deformation, audio, timing and normalization limited |
| 04 | Vegetable grid | Message | Existing captured selection, not a general classifier |
| 05 | Rotate intersection tiles | Message | Existing; other arrangements and exact animation unknown |
| 06 | Tic-tac-toe | Visible transition, incremental A602 | Observed empty-board refresh and winning route; seeded tactical candidate fits recorded replies, original AI/distribution/timing still inferred |
| 07 | Word search | Message; incremental visible transition | Default preserved; explicit second instance with 11 unique selected cells and one shared crossing cell; random generator unknown |
| 08 | License plate | Message; incremental visible transition | Default preserved; separate new photo accepts captured JHB007; new success cannot replace old-instance model failures or establish general normalization |
| 09 | Recursive sign selection | Message | Existing; fixed 31-cell mask, broader boundaries unknown |
| 10 | Whack-a-mole | Message | Existing; original spawn/hit timing not measured |
| 11 | Waldo in 25×25 grid | Message, A402 | Implemented captured photo/selection; local model attempt did not solve; boundary uncertainty retained |
| 12 | Chihuahua/muffin 4×4 grid | Message, A301 | Implemented from successful screenshot tile crops, not unrelated partial image bundle |
| 13 | Select cells without traffic lights | Message, A301 | Implemented complement selection on matched photo |
| 14 | Checkbox among similar statements | Message, A301 | Implemented all 56 labels and scrolling; loading/distractor reset inferred |
| 15 | Waymo parallel parking | Unconfirmed | Partial; effective held controls, driving physics, collisions and accepted parking pose missing |
| 16 | Rotatable 3D text | Unconfirmed | Partial; accepted transcription, scene geometry, camera and motion parameters missing |
| 17 | Draw a circle to 94% | Unconfirmed | Partial; no successful curve or scoring function; incomplete-circle negative example exists |
| 18 | Replacing fire-hydrant images | Message, A401 | Implemented independent captured photo queues and no-target Verify; replacement timing/mistakes inferred; no global click threshold |
| 19 | Flashlight text | Unconfirmed | Partial; successful transcription, illumination dynamics and matcher unknown; A402 contamination must remain flagged |
| 20 | Describe an inkblot | Message, A302 | Partial; one accepted description does not establish an arbitrary-input text judge |
| 21 | Craft a diamond pickaxe | Message, A401 | Implemented conserving stack/right-click/recipes; extra inventory semantics inferred |
| 22 | Capture nine moving ducks | Message, A302 | Implemented real-time individual capture/return/Verify; speed/hitbox/animation inferred; original DOM-assisted record is not a visual-policy result |
| 23 | Find a couple in a panorama | Unconfirmed | Partial; four cubemap faces only, accepted target and zoom/selection conditions unknown |
| 24 | Four-stage visual exam | Message, A303 | Implemented all four same-attempt stages; fresh local model retry completed all four; normalization/other color instances unverified |
| 25 | Free drawing | Unconfirmed | Partial; drawing controls partly observed, accepted drawing and scoring rule missing |
| 26 | Waymo alternate parking layout | Unconfirmed | Partial; held driving, collision/reset and accepted pose missing |
| 27 | 6×6 cable paths covering board | Unconfirmed | Partial; endpoints known, but complete accepted route and extension/undo/intersection semantics missing |
| 28 | Earn $2,500 trading | Message, A402 | Single-share trades and weighted-average realized-profit accounting reproduce all 38 observed transitions; original price dynamics and exact threshold/open-position Verify boundaries remain unknown |
| 29 | Game-specific “soul” image labels | Message, A301 | Implemented captured selection, not a real-world claim about souls |
| 30 | 3×3 sliding sign puzzle | Message, A301 | Implemented adjacent swaps and solved-state evaluator, not an action counter; animation timing inferred |
| 31 | Traffic-light 4×4 grid | Message, A501 | Implemented matched photo and captured 14-cell selection |
| 32 | 5×5 rhythm pads | Unconfirmed | Partial; acceptable pattern, rhythm, audio and success rule missing |
| 33 | Brand-logo characters | Message, A501 | Implemented same-instance screenshot glyph crop; original normalization unverified |
| 34 | Sort mathematical expressions | Message, A501 | Implemented visible expressions and numerical-order evaluator; deselection/reset inferred |
| 35 | Track a ball under shuffled cups | Unconfirmed | Partial; no complete reveal/shuffle/choice/three-round success record; continuous motion missing |
| 36 | Match-three in limited moves | Message, A501 | Initial 8×8 board and 25 swaps audited; clean three-match +30 and four-match +60 observed; full chain/refill/scorer and exhausted-move behavior unknown |
| 37 | Select AI-generated faces | Unconfirmed | Partial; nine photos do not establish the accepted subset; candidate selections are not ground truth |
| 38 | Waymo obstacle parking | Unconfirmed | Partial; short inputs gave no established movement; held controls/physics/parking acceptance missing |
| 39 | Happiness/camera prompt | Message, A501 | Implemented only observed unavailable-camera Verify path, still 0% HAPPY; no media request or expression recognition |
| 40 | Rotating text rollers | Unconfirmed | Partial; accepted text and roller control/timing unknown; disclosed DOM information is not clean visual evidence |
| 41 | Graveside interactions | Unconfirmed | Incremental close-up brushing, return, candle flame and bouquet observed; Verify stays disabled, cleaning threshold and accepted sequence/completion still missing |
| 42 | Conversation/human-confidence meter | Message, A501 | Partial; UI and one transcript known, arbitrary-response scoring/backend unknown |
| 43 | 3D furniture assembly | Unconfirmed | Partial; wood texture is not a furniture model; selecting parts, joining/snapping and completion unknown |
| 44 | Chess against Deep Blue | Unconfirmed | Partial; legal 42-ply middle game only; opponent policy and winning/losing/drawn endings missing |
| 45 | Break up with a fictional AI partner | Rejected/blocked | Partial; failure exchanges exist, accepted dialogue and judging/backend unknown |
| 46 | Empire State Building 64th-floor grid | Message, A501 | Implemented full 14×57 scrolling photo and captured cells; boundary tolerance unverified |
| 47 | Direction-key rhythm dance | Message, A505 | Start mechanism; UI threshold 85% and real failures/success exist, chart/hit windows/synchronization/media missing; incremental download returned 403 and saved zero bytes |
| 48 | Closing video and terminal screen | Visual terminal only, A501 | Partial; normal play-to-Verify transition observed, actual closing video/audio and exact enabling condition missing |

`A402`, for example, means `attempt_402` under that level in P11 or P31.
All levels remain in the delivery scope; readiness is a scheduling decision,
not a reduced definition of success.

## Important interpretation corrections

- Level 28: use the supplemental correction, not the stale original trade count.
  There were 38 BUY/SELL requests (19 each); registration and Verify bring total
  action dispatches to 40 in A402. A fresh raw-log audit now establishes stronger
  accounting evidence; see below. Price-generation rules remain unknown.
- Level 39: the actual final screenshot shows a camera-enable prompt and 0% HAPPY
  while the parent records completion. Reproducing that observed unavailable-camera
  path does not require collecting a user's face and does not implement recognition.
- Level 44: the reviewed black move 10 is `d5f6`; the earlier “extra knight”
  explanation was withdrawn. White move 22 was suggested, not executed. Do not
  manufacture an opponent bug or award success from the middle-game position.
- Level 47: failures A501–A504 stay failures. A505's 87% success used visible DOM
  geometry and keyboard actions; it does not prove screenshot-only Astra success.
- Level 48: a same-attempt `You Are Human` screenshot is terminal evidence for
  that independently opened level. It does not prove levels 1–47 were completed
  in sequence, and the missing completion message must not be fabricated.

## Level 28 accounting evidence update

A second audit paired every trade in
`P11/tasks/level_28/attempts/attempt_402/events.jsonl` with its before/after
visible AX balances. All 19 buys add one share; all 19 sells remove one share.
Starting cash is $500. A weighted-average inventory cost model, with realized
profit updated only on sales and rounded for display, matches **all 38 observed
post-trade profit values**. FIFO and LIFO models disagree with 17 and 18 of the
19 sales respectively. This is strongly supported inference from the captured
instance, not recovery of original source code or every unobserved boundary.

For example, E6→E9 spends $269 for the first share; E47→E50 spends $164 for the
second. E61→E64 sells one for $374, showing profit $158, consistent with
`374 - (269 + 164) / 2 = 157.5` rounded. All 74 held-position AX snapshots have
BUY disabled exactly when cash is less than visible portfolio value divided by
share count. These samples support affordable single-share purchases, not
all-in buys, margin trading or a fixed action-count success rule.

E534 shows profit $2550 with three shares and no validated completion. E555 shows
profit $3134 with zero shares; E558 still has no validated completion 10.27 s
later. E559 clicks Verify, followed by the matched completion record. Thus the
captured instance does not auto-complete merely on exceeding the target, even
after holdings reach zero. Exact $2500, open-position Verify and low-profit
Verify boundaries remain untested.

The older A301 does contain a low-profit Verify probe, not a confirmed negative
example: E29 shows cash $461, profit -$39 and zero shares; E30 requests Verify,
E31 records its return, and the final E33 observation is only about 52 ms later
in the capture logger. It still reports unknown, with no explicit rejection or
completion. No later observation is present. This does not establish a refusal
rule, error message or submission delay; keep that attempt unknown.

There are 324 visible market observations, but their median interval of 461.6 ms
and maximum gap of 50.7 s describe the capture tool, not the market's update
clock. They cannot establish the original RNG, drift, bounds or price process.
Do not advance a recorded price list on each click or embed the winning trade
sequence as gameplay. A separately labeled local price simulator requires an
explicit user choice; no such simulator is implemented or counted as a completed
Neal task by this audit.

Raw event member SHA256:
`9c20aa6c872ca80a9430b9004ff2ce1aedb2109306bc63e70f01e48c07963d37`.
Completion member SHA256:
`da2f744add95dfb01050434fa69ab2e6e5b7ac2a81f664e3aca59c5bc9b4d619`.
Independent calculation and all 38 paired records are retained in
the private reference workspace as `audit_trade.py`
and `level28-accounting-audit.json` beside it. No archived program was executed.

## Level 36 completion and button-state evidence update

A resumed raw-log audit distinguishes the game's victory display from the
explicit task submission. In P31 level 36 / A501, E209 shows score 1260 and six
moves remaining; E211 shows `Sweet Victory!` while the validating parent still
has no completion record. E212 then requests Verify, followed by the saved
same-instance completion message. Do not substitute the victory overlay for
the Verify step or infer a precise cross-clock latency from these records.

All 54 AX snapshots containing Verify describe it as an ordinary button,
including the initial score-zero state; none marks it disabled. This constrains
the visible control reconstruction: the capture does not support disabling
Verify until the target is reached. It does **not** establish what a low-score
submission or an exhausted-move submission would do; neither outcome was
observed in this attempt.

Raw event member:
`P31/tasks/level_36/attempts/attempt_501/events.jsonl`.
SHA256: `86d9405202f7e75339b667301074ff4f59f1d6c3212716ba76ae0c32f8274c25`.
The incremental `offline_supplements/level_36/score_observations.json` restates
this old run rather than adding a new game. Full cascade/refill/scoring behavior
is still missing; this update does not add level 36 to the implemented catalog.

## Reuse the current framework

No new browser-control framework or second static server is needed for the basic
local game. Existing owners already provide:

- [registration.py](/examples/not_a_robot/registration.py): registration from the
  local catalog and direct environment lifecycle.
- [env.py](/examples/not_a_robot/env.py): owned Playwright browser, normalized GUI
  dispatch, mouse down/move/up, key down/up/hold, budgets, observations and cleanup.
- [local_tasks.py](/examples/not_a_robot/local_tasks.py): owned loopback server and
  public task status contract.
- [recorder.py](/examples/not_a_robot/recorder.py) and
  [codex_bridge.py](/examples/not_a_robot/codex_bridge.py): event/image provenance,
  public action summaries, screenshot crops, tool errors and finalized outcomes.
- [codex_smoke.py](/examples/not_a_robot/codex_smoke.py): real isolated Codex attempts.
  Task selection now derives from the implemented reference catalog; use explicit
  `--tasks` for small smoke runs rather than the whole default set.

The reference collector's cross-site iframe drag failure does not establish a
defect in this local top-level Playwright runtime. Existing dispatch support is
not, by itself, proof of sufficient timing for drawing/driving/rhythm games.
Add owned input calibration tests and actual per-task browser tests.

## Implementation sequence for all 48

1. **Evidence and asset import (first expansion implemented).** Extend the private importer to the reviewed
   packages using explicit allowlists and hashes. Match each imported image to
   its actual captured instance; retain authored approximations as such. Do not
   extract the whole outer ZIP or serve work scripts/executables. Unknown artwork
   rights mean original assets remain private and ignored by Git.
2. **First static expansion (implemented).** Implement 11–14, 21, 24, 29–31, 33–34 and 46.
   These twelve new tasks have sufficiently concrete captured-instance paths to
   begin. Keep progress on these independent of the unresolved first-ten details.
   Test real input controls, wrong selections, reset and final verification, not
   just image display or registration count.
3. **Real-time and multi-action games (18/22 implemented with limits).** Build the needed mechanics for 15, 17–19,
   22, 25–28, 32, 35–36, 38 and 47. Several components can proceed now, but exact
   scorers/dynamics must wait for the missing observations. Calibrate held keys,
   continuous mouse paths and batch timing. Never freeze the game during model
   inference or stretch hit windows merely to obtain a pass. Current local
   parameters and remaining evidence are in [DYNAMIC.md](/examples/not_a_robot/DYNAMIC.md).
4. **Spatial, narrative and opponent tasks.** Implement 16, 23, 40–41, 43–44
   against additional scene, interaction and terminal evidence. A different chess
   engine or invented furniture snapping rule is an approximation, not recovered
   original behavior. Resolve the existing level-six opponent gap here too.
5. **Judging and media tasks (39 unavailable-camera branch implemented).** Resolve 20, 39, 42, 45 and 48. The three text/dialogue
   tasks need a defensible arbitrary-input acceptance model; replaying one transcript
   or awarding fixed points per message is not an interactive reconstruction.
   An explicitly approved substitute LLM judge would need a selected local model
   or API credentials and must be labeled as a substitute. The archive does not
   establish the original backend or prove that an API is mandatory. Obtain lawful
   media files for 47/48; reproduce only the evidenced camera path unless a broader
   camera workflow is separately specified and authorized.
6. **Campaign and full acceptance.** Add local 1→48 transitions, reset/retry and
   task-boundary events while retaining independent task entry. Run scripted
   environment checks and real screenshot-only Codex attempts for every task;
   retain failures and retries. Verify the campaign itself instead of jumping
   directly to 48. Completing the environment does not guarantee every model run
   will solve every task.

Likely example-only changes:

| Owner | Required extension |
| --- | --- |
| `reference_assets.py` | Multi-package allowlist, per-level asset/instance provenance; no broad ZIP extraction |
| `local/tasks.json` | All implemented tasks, observed/inferred/unknown rule metadata; no empty placeholders counted as tasks |
| `local/neal.js`, `local/neal.css` | Task implementations; split real rendering/mechanic owners into small local modules as the ten-task dispatcher grows |
| `local/index.html`, `local/game.js` | Load task modules and retain independent task previews; environment-owned campaign navigation must preserve terminal frames before page unload |
| `local_tasks.py` | Explicit asset routes/MIME/CSP for new modules and lawful media; avoid copying large video payloads into every environment's in-memory asset map |
| `env.py`, `registration.py`, `evaluator.py` | Owned local campaign navigation/state, real-time input coverage and genuinely required observed completion modes; local and remote campaign IDs stay separate |
| `recorder.py`, `codex_bridge.py` | Preserve action/observation provenance across new interactions; optional genuine owned-browser video, never stitched screenshots presented as video |
| `codex_smoke.py`, tests | Catalog-derived supported tasks, task-specific bounded budgets and per-level/campaign regressions |

Core CUA-Lite, Slime and the top-level training pipeline do not need changes just
to build these local pages. Container packaging can wrap the same owned runtime
later; it is not a replacement for missing game rules. Canonical training export
and production env-server pooling remain separate integration work.

## Smallest useful supplemental capture request

Prioritize successful, normal-UI runs for the 17 unconfirmed levels listed above.
For dynamic tasks, include the actual continuous gameplay video with the relevant
audio when authorized, not only before/after screenshots. Keep a few genuine
negative examples where the boundary determines the local scorer.

In particular:

- 6: additional opponent games, measured reply timing and official completion
  event; the normal refresh-to-win flow is now captured. 8: unspaced/spaced
  acceptance/rejection on the same pictured plate; the new JHB007 instance does
  not establish normalization for the old 867V 309 instance.
- 15/26/38: sustained driving, collisions/reset and the accepted parking pose.
- 16/19/40: complete successful transcription and the visible movement/lighting.
- 17/25/27: successful continuous paths/drawings, actual resulting score and undo.
- 23: full panorama and the accepted target/zoom; 43: actual assembly gestures
  and geometry references; 41: the full enabling/completion interaction.
- 32/35/47: continuous sequence/timing and success; 37: a genuinely accepted
  selection; 44: a complete legal winning game and reset/opponent behavior.
- 20/42/45: more than one input/output path, genuine rejection/acceptance, and
  authorized backend specification if exact behavior is required.
- 36: dense simple-three/four/five-match, crossing and per-wave cascade/refill
  evidence, plus exhausted-move/low-score Verify. The captured 25 swaps include
  one invalid swap that restores the board without spending a move; final
  900→1260 does not establish a universal cascade formula.
- 47/48: the actual relevant media, with rights
  and source information. No face video is required for the observed 39 path.

Do not request cookies, authentication tokens, browser profiles, raw HAR files
or bypasses of real access controls. A single successful run is useful evidence,
not a proof of every random variant or acceptance boundary.

## Completion gates

Each task needs an actual interactive page, normal GUI control, a defensible
state transition and evaluator, instance/asset provenance, reset/error behavior,
and archived end-to-end tests. Track separately:

1. implemented mechanism;
2. reference fidelity and remaining unknowns;
3. scripted environment correctness;
4. real model attempt outcome;
5. recording and cleanup completeness.

All 48 task gates and the sequential campaign gate must be checked before calling
the full request complete. The current expansions leave 23 task implementations
and full campaign acceptance unfinished, alongside the recorded fidelity gaps of existing
tasks. Tests and actual model outcomes are tracked separately in
[validation](/examples/not_a_robot/VALIDATION.md). Original artwork and run
artifacts remain private; publishing the implementation does not resolve the
remaining task or fidelity gaps.
