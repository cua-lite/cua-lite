# Local reconstruction validation record

## Publication verification, 0.6.0 (2026-09-15)

This publication includes 25 captured-instance Neal task paths, the local
campaign controller and explicit incremental 07/08 instances on
`experiment/neal-first10-local`. It is not an all-48 implementation; level 39
still covers only the recorded unavailable-camera branch.

A fresh full example-suite invocation, including the local-browser tests,
passed **265 cases and two subtests in 456.11 seconds**. Independent counting of
the JUnit records found 265 distinct cases with no failures, errors or skips.
Separate non-browser runs before and after formatting each passed 161 cases
and two subtests. The five formatted test files retained identical Python ASTs;
no game logic, acceptance rules or model prompts changed for this publication.
Ruff lint, formatting checks on all 21 changed Python files, all eight local
JavaScript syntax checks and Git whitespace checks passed.

Reports are private operator artifacts under:
`/mnt/weka/shrd/k2m/lingjie.chen/neal-publish-validation-20260915.m7B6q5/`.
The full report is `full.xml`, SHA256
`0058b0c6c4f82f3940e372ac7d649a52ff0e0ce4d7a702db0afe911ab8322711`.

All 54 private reference-image hashes and Git ignore rules were checked. Only
the example's source, documentation, tests and reference manifest are included;
original artwork, reference ZIPs, model trajectories and caches remain outside
the commit. No new model attempt or original-site interaction was performed.
The earlier 22/25 result and later 3/3 instance-specific result remain separate.
The dated sections below describe their historical code, preview and Git states,
not the current remote branch or a currently running preview.

## Incremental reference refinement, 0.6.0 (2026-09-11)

The newly supplied incremental archive supports the scoped changes described in
[INCREMENTAL.md](/examples/not_a_robot/INCREMENTAL.md): evidence-based level-six
opponent refinement, explicit additional 07/08 instances, and corrected evidence
limits. It does not add implemented levels or replace any historical score.

Three fresh, independent screenshot-only Codex attempts all succeeded:

| Task | Explicit instance | Actual outcome | Attempt duration | Client exchanges / GUI primitives |
| --- | --- | --- | ---: | ---: |
| 06 Tic-tac-toe | default, refined opponent | Lost first board; refreshed once, won with a real X column and Verify | 143.65 s | 14 / 28 |
| 07 Word search | incremental | Selected 11 unique cells, including the shared crossing cell; first Verify accepted | 35.54 s | 4 / 24 |
| 08 License plate | incremental | One rejected spaced input, then captured exact unspaced input accepted | 38.18 s | 6 / 11 |

The run used frozen 0.6.0 code/assets, `gpt-6-astra` / `xhigh`, seed 0,
600 seconds and 150 environment steps per task, in two lanes. Only the existing
three GUI MCP tools were exposed; no DOM/source/shell/web input or answer hints
were supplied. There were no automatic episode retries. The first losing board
in 06 and rejected submission in 08 remain inside their recorded attempts.
Public action summaries are retained; private reasoning was not read or exported.

The final audit verified **467 events, 73 per-attempt PNGs, 63 paired GUI
dispatches and 24 matched client exchanges**, including model-delivered image
hashes, public decision-to-observation links, crops, seed/instance/archive
identity and success before finish claims. All 896 frozen source/private-asset
files and 66 served asset hashes matched. There were no extra client calls,
display truncations or recorded infrastructure errors. All three CLI runs exited
zero, recordings finalized, listeners closed, and all 36 sampled owned process
identities were absent afterward. Process sampling is not an inventory of every
browser process ever started. Client evidence is not provider-side attestation.

This is **3/3 on these specific refined instances**, not a new all-25 score or
all-48 result. New 08 success does not replace the old `867V 309` failures. Local
rejection of `JHB 007` does not prove the original site rejects it. The original
AI/generator, timings, media and unresolved task rules remain evidence gaps.

Artifacts:
`/mnt/weka/shrd/k2m/lingjie.chen/neal-refine-incremental-20260911.44nAGE/`

```text
plan.json
before-example.tar.gz
before-tracked.patch
code-snapshot.tar.gz
private-assets.tar.gz
preflight.json
lane-a/3bc56c20883a49559a4ed26cef8f9ed2/run_manifest.json
lane-b/3d943eb148444b3284658846ae45d8d3/run_manifest.json
model-audit.json
process-check.json
```

Model-audit SHA256:
`c2a83abb04c811266788e294f9f38bb0cc57336e7cf03a2c3698858d11b33433`.
The earlier all-25 audit hash below was independently rechecked unchanged.
No commit, push, global configuration change or original-site request was made.

Regression coverage reconciles to **265 distinct passing test cases**, plus the
two subtests reported by pytest. This is the completed full sweep plus explicit
reruns, not a claim of one all-green full-suite invocation:

- `full.xml`: 250 passed, 14 failed. All 14 failures were old test expectations
  missing the newly recorded `reference_instance` field, not failed gameplay.
- `unit-final.xml`: all 161 non-browser cases passed after updating the six
  affected observation-fixture expectations; two subtests also passed.
- `authored-final.xml`: all eight authored-game GUI cases passed after updating
  their exact state-key expectation, retaining the gameplay and cleanup checks.
- `default-grid.xml`: the added full-100-character default word-search regression
  passed. Existing default accepted-set, incremental shared-cell, seed/reset,
  win/draw/loss, plate-isolation and real 01–14 campaign regressions also passed.

`completion-check.json` reconciles the exact collected test identities against
these reports with no unresolved failure or missing case. Ruff, formatting,
Node syntax and Git whitespace checks passed. After model freeze, only docs and
tests changed; runtime code and private asset bytes remained identical. All prior
example files remain present. The owned preview was gracefully restarted on the
same loopback port 46323; version 0.6.0, 33 catalog entries and all 66 served asset
hashes were checked. No other service was stopped.

## Earlier fresh Codex run on all 25 paths, 0.5.0 (2026-09-11)

All 25 currently implemented captured-instance paths received **one fresh actual
Codex attempt each**: **22 success, three `agent_stopped` failures** (08, 09, 11).
All attempts are evaluated, recording/cleanup-complete and from distinct model
threads and environment attempts. This is independent-task testing, not an
all-48 campaign, and previous development successes do not replace fresh failures.

The frozen run used `gpt-6-astra`, `xhigh`, seed 0, a 600-second controller budget
and 150 environment steps per task, in two serial lanes. A step can contain
several mouse/keyboard primitives. The deadline includes controller startup;
reported environment durations begin at reset. Codex operated through screenshots
and the three GUI MCP tools, with a read-only solver workspace and shell, web,
apps, memory and multi-agent tools disabled. No answers were supplied to the
solver, no automatic outcome retries were added, and no game rules were changed.

Successful levels: **01–07, 10, 12–14, 18, 21–22, 24, 29–31, 33–34, 39, 46**.
The previously untested eight (12, 13, 14, 29, 30, 31, 33, 46) all succeeded.

| Failed level | Environment time | Rejected submissions | Observed result |
| --- | ---: | ---: | --- |
| 08 License plate | 193.11 s | 10 | Never submitted the local exact string `867V 309`; stopped voluntarily |
| 09 Recursive stop sign | 490.27 s | 12 | Never submitted the exact 31-cell selection; stopped voluntarily |
| 11 Find Waldo | 587.64 s | 9 | Never submitted all three required cells together; stopped near the budget limit |

These three are model non-successes, not launcher timeouts or infrastructure
interruptions. In particular, 08's original-site whitespace normalization remains
unverified: the failure does not establish that the model failed to recognize
the characters. No matcher or selection boundary was weakened to improve scores.

The final audit verified **9,215 events, 1,481 per-attempt content-addressed PNGs,
1,492 paired GUI input dispatches and 312 matched client exchanges**. PNG counts
include observations and crops and are not necessarily globally unique. Checks
cover event/image bytes and hashes, monotonic sequences/timestamps, public
decision-to-observation links, client input image/feedback delivery, pixel-exact
crop derivation, actual model/tool/budget configuration, and game success before
model finish claims. All 893 frozen code/private-asset files matched the source
archives; all 65 served asset hashes matched the preflight. Private reasoning was
not read or exported; client records are not provider-side model attestation.

Five CLI display truncations occurred (05 turns 2/5, 11 turns 2/38, 46 turn 9).
Matching typed client session outputs retain the full expected images, with
verified hashes. The audit records these display exceptions explicitly. There
were no extra client calls or recorded environment error events in this run.
All 68 game rejection events and six refreshes remain in the trajectories.
Rejections include 31 missed mole clicks in level 10, not only form submissions.

Both complete lane manifests agree with their per-task results and exited zero;
zero means orchestration completed, not universal task success. All 25 recorded
listener ports were checked closed, both lane tmux sessions ended, and none of
the 69 observed scoped process IDs remained. That process list is a sampled
descendant inventory, not an inventory of every browser ever started. Every
attempt separately reports successful owned-resource cleanup. The independent
preview at loopback port 46323 remained live with its 33-entry catalog.

Artifacts under
`/mnt/weka/shrd/k2m/lingjie.chen/neal-codex-all25-20260911.VVwSmT/`:

```text
SUMMARY.md
plan.json
preflight.json
code-snapshot.tar.gz
private-assets.tar.gz
lane-a/33a7061feead4f2b8bf7eb12df75d906/run_manifest.json
lane-b/e56aaceb80554348be2ecb14baf9bd5c/run_manifest.json
model-audit.json
completion-check.json
process-check.json
```

Final model-audit SHA256:
`458536b962dcdef3abbb098f4b7216b9d7dfe094e74c205cfd483df81cd43dcc`.
This validates the frozen local instances only: 06 still has an inferred local
opponent, 39 is only the unavailable-camera branch, 23 levels remain unimplemented,
and neither original-site parity nor all-48 campaign completion is established.
Only result documentation was updated after the run; no runtime changes, staging,
commit or push were made for this testing increment.

## Real 01–14 campaign prefix and actual missing level 15 (2026-09-11)

A new GUI-oracle regression traverses the real implemented levels 01 through 14
in one campaign, then reaches the **actual absent `neal_15`**, without injecting
a missing catalog entry, substituting page state, skipping a level or declaring
an early finish. The game/evaluator/runtime code is unchanged. This test uses
known reference answers and visible DOM geometry, so it is **not model gameplay**.
In particular, level 06 exercises its existing inferred local opponent, not a
newly verified original-site winning policy.

The final attempt completed all 14 per-task evaluators and stopped with
`unsupported_task`, `truncated=true`, `terminated=false`, and campaign reward 0.
The display remains the successful level-14 page; feedback names level 15 as
missing and retains the total requirement of 48. Further `step()` calls are
rejected. One browser, context, page, server, recorder, resource ID, global step
counter and elapsed-time budget persisted throughout. No preview service was
stopped or reused for the test.

The final archive contains **1,495 events, 159 content-addressed PNGs, 278
dispatched GUI primitives and 98 environment steps**, over 35.64 seconds. The
audit matched event/image bytes and hashes, per-stage event sequences, global
UTC/elapsed timestamps, terminal screenshots before next-page start, all 65
served runtime asset hashes and all 43 non-Markdown snapshot files. There are
14 ordered stage starts/completions, one browser start and no model-decision
events. Recording and owned-resource cleanup both completed.

The first new check passed in 37.16 seconds. Read-only review subsequently found
a potential test timing flake: if the checkbox timer finishes during the click's
screenshot, an unconditional separate wait can arrive after the campaign has
already stopped. The test now submits click and wait in one canonical batch,
allowing the existing boundary logic to discard an unnecessary tail. No game
timer or production boundary behavior changed. The final **20 campaign tests
passed in 66.06 seconds**. Ruff and diff checks passed. The full example suite
now collects 247 tests; it was not rerun in full for this test-only increment.

Artifacts under `/mnt/weka/home/lingjie.chen/.tmp/neal-prefix14.ybIWGg/`:

```text
prefix.xml
initial-test-source.tar.gz
campaign.xml
final-test-source.tar.gz
final-pytest/test_real_first_fourteen_stage0/b03a8162f89d44c18654d4906ef0b9bd/
audit_prefix.py
prefix-audit.json
```

Final campaign JUnit SHA256:
`0935783c46ab32c60da1fb88b0a008696583cc9c0ef0f07ca0fa671f33d3c28d`.
Final source snapshot SHA256:
`e5c49e8856621c3c36f3e22528d80200158a01c812f7857497139efd4507b708`.
Attempt event SHA256:
`74fac732d3d8e28bd076df29e9c385a6e7e77b0c9306ee9374c682a8dcc09bc2`.

This closes the earlier gap between an injected missing-task test and the real
catalog boundary. It does not implement level 15, prove model success on 14
levels, resolve reference-fidelity gaps, or complete the 48-level campaign.
No new reference package, model invocation, dataset, commit or push was involved.

## Additional model coverage and client-image audit (2026-09-11)

Two task targets were attempted with the unchanged post-campaign-fix runtime:
`gpt-6-astra`, `xhigh`, seed 0, 180 seconds and 60 steps per fresh attempt.
Level 24's first attempt was interrupted by the launcher; only that task was
restarted. The interrupted archive remains intact and is not a model score.

| Attempt | Actual outcome | Controller exchanges | GUI primitives | Environment time |
| --- | --- | --- | --- | --- |
| 11 Find Waldo | `agent_stopped`, not solved | 21 | 33 | 172.37 s |
| 24 Visual exam, initial | `controller_disconnected`, not evaluated | 7 | 10 | 52.25 s |
| 24 Visual exam, fresh retry | `success`, four stages completed | 8 | 19 | 40.75 s |

Level 11 made four rejected Verify submissions and one refresh. Its failure
remains a failure; neither the selection mask nor visual difficulty was changed.
The interrupted level-24 attempt completed stages 1 and 2 only. Its retry
completed all four stages in order with zero rejected submissions. The game's
success event precedes the model's finish claim. This establishes one local
captured-instance success, not original-site parity or an all-task success rate.

The three attempts contain **614 events and 94 per-attempt content-addressed
PNGs**. Independent checks matched event/image hashes and sizes, event sequences,
elapsed timestamps, runtime assets, public decision-to-observation links, all 36
controller exchanges, and terminal/cleanup records. Matching client session tool
outputs contain the expected full screenshots and crops. Only tool-call/output
envelopes and client configuration were inspected; private reasoning was not
read or exported. Client evidence is not provider-side cryptographic attestation.

One level-11 CLI MCP display event contains a nested JSON text result with the
marker `481017 chars truncated`. It is **not a usable image archive**. The
initial audit therefore failed instead of treating truncated base64 as a PNG.
Matching the original request's `itemId` to its client session tool call and
output establishes that both typed input images were retained there. Their
SHA256 values exactly match the full screenshot and 1600×1600 crop in the
environment archive. This was CLI display truncation, not evidence of image
delivery loss. The audit records this exception explicitly; no compatibility
shim, image compression, global Codex setting or runtime change was needed.
The [official tool-event documentation](https://learn.chatgpt.com/docs/app-server#items)
was consulted for event structure; the truncation finding comes from the local
matching records, not a documented universal size limit.

Artifacts under `/mnt/weka/home/lingjie.chen/.tmp/neal-model-coverage.Yr2Fqz/`:

```text
source.tar.gz
source.patch
model/d36e42bdc3324aeebf8786d38921ac3d/neal_11/
model/d36e42bdc3324aeebf8786d38921ac3d/neal_24/
retry24/45085cfee49f4173933e254a06579027/run_manifest.json
retry24/45085cfee49f4173933e254a06579027/neal_24/
audit_attempts.py
model-audit.json
```

The interrupted initial group has no `run_manifest.json`; its per-attempt
manifests and interruption result are retained without fabricating a completed
group. Audit SHA256:
`e68a8b017ac592654d85cf4732d70d403f45fc4fe41e30845a90ae6f343eb75d`.
Frozen source snapshot SHA256:
`df3f025101e890ad6820321e2e8af7340157157233109729f1e55eab6a423304`.
All recorded runtime assets still match the current worktree. No production code
changed in this coverage/audit increment; the 246-example/630-selected-shared
regression below is the existing post-fix validation, not a newly rerun suite.

The old preview process was confirmed absent and its port unbound before being
restored in owned tmux session `cua-neal-preview-46323-Yr2Fqz`, on host
`fs-mbz-login-big-001`, at `http://127.0.0.1:46323/`. A separate check confirmed a
live pane, server process and 33-entry catalog (25 reference paths plus eight
authored exercises). This is a point-in-time launch check, not continuous uptime
monitoring. SSH forwarding is still required for remote laptop access; the
service remains loopback-only. No container, real CAPTCHA, dataset generation,
staging, commit or push was involved.

## Ordered local campaign controller (2026-09-11)

The existing 0.5.0 task pages now have a separate local campaign environment,
`visual_tasks_campaign@full_game`, and explicit `--campaign` bridge/smoke entry.
No task rules, speeds, artwork or success thresholds changed in this increment.
The catalog remains 25 implemented reference paths; full all-48 gameplay and
campaign acceptance remain incomplete. See [CAMPAIGN.md](/examples/not_a_robot/CAMPAIGN.md).

One real screenshot-only Codex attempt requested `gpt-6-astra`, `xhigh`, a
120-second controller budget and 12 environment steps. It **completed levels
01, 02, 03 and 04 in order, entered 05, then stopped with `controller_timeout`**.
This is a failed/budget-limited full-campaign attempt, not full-game success or
four independent reset successes. Browser, context, resource ID, recorder and
budget persisted across all four transitions. Environment elapsed time was
113.05 seconds because controller startup preceded the environment's reset.

The audit verified **348 events, 59 content-addressed PNGs and 46 dispatched GUI
primitives**, including per-page event sequences, terminal-frame-to-next-page
ordering, global timestamps, completed-prefix accounting, source bytes, public
decision/image/crop links and cleanup. There were 12 completed controller-to-env
exchanges. The outer client also retained two later failed MCP calls (`computer`
and `finish`) with `Transport closed` after the deadline had closed the bridge.
Those two calls never reached the environment and were not silently counted as
executed actions. No finish claim was used to award campaign success.

The attempt manifest independently reports `recording_complete=true`,
`cleanup_complete=true`, `outcome=controller_timeout`; the launcher reports
`success=false`. An evaluated attempt or a zero launcher exit code does not mean
the game was solved. This test did not reach the current real missing level 15.
The missing-next boundary is instead tested with an explicitly injected absent
level 02, and final48 reward accounting with a clearly labeled controller-only
fixture, not fabricated gameplay.

Targeted verification:

- Final post-fix regression: **all 246 collected example tests passed**, split
  into two disjoint processes (62 in 222.20 s; 184 in 217.20 s, plus two subtests).
  The verification script matched individual JUnit testcase identities against
  actual collection: no missing, duplicate, failed or skipped tests. Source and
  test hashes matched the frozen post-fix snapshot. The earlier whole-suite run
  retained the two already-described test-harness failures; one short rerun was
  deliberately interrupted when review found the cancellation grading defect.
  Neither is substituted for these final passing reports.
- The original 14 campaign checks passed in 34.05 s, including genuine 01→02 progression,
  timer completion during a no-tool interval, unexecuted batch tails, one global
  budget, held-key/button release, reset/finish, absent-next and navigation-error
  outcomes. Early test-only fixes corrected a manifest field path and a callback
  wrapper. The held-input DOM probe also needed ordinary page background: the
  completed checkbox disables itself and suppresses mouse events. Final probing
  confirmed the right button remained held after a left click (`buttons=2`) and
  received its trusted release on the old page before navigation.
- Subsequent review reproduced a cancellation-classification defect: a controller
  deadline during next-page navigation was being labeled `infra_error`. The
  transition and observation owners now propagate cancellation for the controller
  to classify, preventing a half-navigated environment from being stepped again.
  Ordinary navigation errors still remain infrastructure failures. The real model
  attempt above predates this fix and did not trigger it; its original source
  snapshot and audit are retained rather than relabeled as a post-fix model run.
  Five cancellation regressions then covered navigation, state reads, ordinary
  screenshots, completed-page screenshots and final-stage archive screenshots.
  They also exposed and fixed a provisional per-page success assignment that
  could otherwise survive a cancelled final screenshot as false campaign success.
  The final **19 campaign checks passed in 28.96 s**, verifying actual bridge
  finalization as a timeout and rejecting further steps on a cancelled environment.
- 70 bridge/launcher checks passed in 6.53 s, including a real-browser 01→02
  screenshot/crop-provenance test, explicit campaign CLI selection and excluding
  missing-task stops from evaluated model outcomes.
- 630 selected shared action/geometry/feedback/metadata tests passed in 2.20 s.
  This is not a full-core, container or production env-server validation.

Artifacts under `/mnt/weka/home/lingjie.chen/.tmp/neal-campaign.IHGpHg/`:

```text
model/a0515a4720a4468ebe7d1ffcafdf06d7/run_manifest.json
model/a0515a4720a4468ebe7d1ffcafdf06d7/full_game/
model-audit.json
audit_model.py
bridge.xml
shared-selected.xml
final-a.xml
final-b.xml
final-collected.txt
final-verification.json
verify_final.py
pre-model-source.tar.gz
pre-model-tracked.patch
post-fix-source.tar.gz
post-fix-tracked.patch
```

Model audit SHA256:
`031fd9247c7815d9143c6d25d19b2bac5e826960b4660e1fdea9931e2b267cc6`.
Selected shared report SHA256:
`a7a9f9b5e47640e8d27f5e2fe76026d0d75cdb0ed20579b7802e710155486893`.
Final verification report SHA256:
`ff4e1bab5558dff27b0d554253cabaeafcbbe0cf7a1d16f1a52b470a5d2e28c5`.
Post-fix source snapshot SHA256:
`3276e26ea0e60536f5db544ff46d7d716d834619163a49aee9e7e2052ad5ea62`.
The separate final 19-case report is
`/mnt/weka/home/lingjie.chen/.tmp/neal-campaign-tests.WqrKJt/final19.xml`, SHA256
`398d9996577c22785fbfe058639843b128221d2fd3e8ec273e4cd1a35a187564`.

No remote website, credential changes, private reasoning export, core CUA-Lite
changes, dataset generation, staging, commit or push were involved. The preview
continues serving independent tasks; the model campaign owns its own temporary
listener and browser, both now closed.

## Dynamic and unavailable-camera expansion (2026-09-11)

Version 0.5.0 adds levels 18 and 22 plus **only** level 39's observed unavailable-
camera branch. The catalog now contains 25 reference task paths and eight separate
authored exercises. The other 23 task implementations and the continuous campaign
remain unfinished. This expansion reuses the existing example-only owners and
does not implement facial recognition, recover original random generators or
claim original timing equivalence. See [DYNAMIC.md](/examples/not_a_robot/DYNAMIC.md).

The same reviewed expansion ZIP now imports 38 private images (11 newly added),
giving 53 allowlisted reference images including the first-ten supplement.
All 53 member hashes and ignored status were checked. Level 18 uses nine unchanged
same-attempt screenshots as per-cell photo queues. Level 22 uses the original
transparent duck sheet/net. Level 39 needs no image or media asset.

Final engineering validation:

- **223 example tests passed in 371.44 s**, including four replacement tests,
  three duck tests, three camera-path tests, two continuous-input calibration
  tests and the new limited-camera metadata check. There are 223 individual
  JUnit testcase records with no failure/error/skip; the aggregate `tests=225`
  is not used as the count.
- **630 selected shared action/geometry/feedback/metadata tests passed in 2.18 s**.
  This remains a selected suite, not a full-core or container integration claim.
- The first pointer calibration crossed selectable page text and entered native
  text drag-and-drop, suppressing intermediate mousemove events despite all 20
  dispatches being recorded. Calibration moved to verified blank space, adding
  an explicit no-native-drag check. Both final calibration cases passed; no
  production input code was changed. Future drawing surfaces must manage text
  selection/native drag appropriately.
- Keyboard holds persisted across observations and a no-tool interval while
  requestAnimationFrame advanced; key release stopped the held state. Separate
  mouse down/move/up and the 20-segment canonical drag were delivered as trusted
  browser events and recorded. This does not establish original vehicle physics
  or drawing/rhythm scoring.
- Camera tests instrumented a fresh document before scripts and Verify: zero
  device/media requests, no camera grant, zero displayed happiness both before
  and after the observed-branch completion. Actual UI screenshots were checked.
- Ruff, new JavaScript syntax, documentation links and `git diff --check` passed.

Three fresh screenshot-only Codex attempts requested `gpt-6-astra`, `xhigh`,
seed 0, 300 seconds and 100 steps each. Code and dynamic parameters were frozen
before these runs; there was no outcome-based speed, hitbox or rule adjustment.

| Task | Actual outcome | Model tool calls | GUI primitives | Attempt duration |
| --- | --- | --- | --- | --- |
| 18 Replacing hydrants | Success | 12 | 70 | 86.45 s |
| 22 Roaming ducks | Success | 17 | 220 | 111.29 s |
| 39 Unavailable-camera path | Success on this branch only | 4 | 5 | 18.96 s |

The level-18 archive contains 30 actual independent image replacements, with
650 ms delays, and no hydrants at final verification. Level 22 caught nine distinct
entities with no refresh. It also logged **25 missed background clicks and nine
ignored clicks on returning/caught ducks**. The task's `mistakes=0` counts rejected
Verify submissions, not perfect aiming; it is not a hit-rate score. The controller
records exploratory drags as well as ordinary clicks and waits.

The three attempts' **1,301 events and 190 per-attempt content-addressed PNGs**
passed independent event/image hash, sequence, timestamp, image/crop provenance,
decision-link, actual MCP tool-name, client model/effort/workspace, source-byte and
cleanup audits. The audit separately checks the final replacement/capture/device
branch events. No DOM/source/evaluator answers were available to the model, no
private reasoning was requested/exported, and no original website was contacted.
These three local successes do not establish all-25 model success or all-48 parity.

Artifacts under `/mnt/weka/home/lingjie.chen/.tmp/neal-dynamic.h2wkJI/`:

```text
model/7c6f073935ca49fa863f607be6263963/run_manifest.json
model/7c6f073935ca49fa863f607be6263963/neal_18/
model/7c6f073935ca49fa863f607be6263963/neal_22/
model/7c6f073935ca49fa863f607be6263963/neal_39/
model-audit.json
audit_model.py
regression.xml
shared-selected.xml
calibration.xml
calibration-retry.xml
preview2-gallery.png
preview2-neal_18.png
preview2-neal_22.png
preview2-neal_39.png
```

Model audit SHA256:
`717edddf0d571f1b6a7b0130288f2ed6f304d1afd30abdcfa3641970e444e75c`.
Example report SHA256:
`15509b904d65e2492a837945ab42aac9e71b634ecc284256bdecfc34fc37cc18`.
Selected shared report SHA256:
`6647c991273c47c775951e533ebbf2276800a7d63a024962d19d9c223cef6e7a`.

The owned preview was updated on the same loopback port and verified with all
three new direct URLs plus the 33-card gallery. An initial probe's string-eval
wait helper hit CSP; using a visible DOM locator fixed the probe without changing
the page's security policy. The passing screenshots are the `preview2-*` files.
All model/test browsers were closed. The explicitly launched preview remains
running for user access. No staging, commit or push was performed.

## First twelve-task expansion (2026-09-11)

Version 0.4.0 adds levels 11–14, 21, 24, 29–31, 33–34 and 46: 22 implemented
captured-instance tasks in total, plus eight separate authored exercises. The
work was uncommitted on `experiment/neal-first10-local` at base `2ecf495` when
this validation was recorded.
Core CUA-Lite and Slime are unchanged. See
[support and private-import instructions](/examples/not_a_robot/SUPPORT.md).

The importer now verifies either reviewed ZIP and its exact allowlisted members;
15 first-ten files plus 27 expansion files are imported privately. Tests cover
idempotence, mismatched archives/member hashes, preserving different user bytes,
symlink refusal, independent imports and serving a partial import without hiding
the missing-assets error for affected tasks.

Browser tests exercise actual canonical GUI inputs: precise selections and wrong
submissions, 56-card scrolling, four successive exam stages, inventory material
conservation and right-click placement, legal sliding plus reversible detours,
full numerical ordering, long-image scrolling, refresh and episode cleanup.
Initial scoped runs found test-only selector/visibility mistakes. Visual review
also caught a real MathML π missing glyph despite passing logic tests; the
renderer now uses an explicit normal π glyph and DejaVu Serif. The corrected
board suite passed eight cases and its actual screenshots were inspected.

The first expansion's real Codex smoke uses two fresh screenshot-only attempts,
the existing `gpt-6-astra`/`xhigh` request, seed 0, 300 seconds and 100 steps each:

| Task | Actual outcome | Model tool calls | Recorded GUI primitives | Attempt duration |
| --- | --- | --- | --- | --- |
| 21 Craft a pickaxe | Success | 8 | 44 | 50.90 s |
| 34 Math ordering | Success | 5 | 20 | 28.02 s |

Both had zero rejected GUI calls and zero task mistakes. Their 373 events and
67 per-attempt content-addressed PNGs passed an independent audit of event/image
hashes, sequences, timestamps, actual MCP tool names, screenshot/crop provenance,
decision-to-observation links, client model/effort/workspace configuration and
final cleanup. Both attempts' recorded source assets match the corrected current
bundle; the earlier board-test source snapshot is retained separately. Client
configuration evidence is not cryptographic attestation of provider internals.

These two successes establish neither model success on the other new tasks nor
original-site equivalence. The model had no DOM, source, selectors, evaluator
answers, shell, web or prior trajectories. No original website was contacted.
Scripted reference-oracle tests and model attempts remain distinct evidence.

Artifacts in `/mnt/weka/home/lingjie.chen/.tmp/neal-expansion.fNOwY9/`:

```text
model/82fe03aaa71945a18fbcba569329d8e7/run_manifest.json
model/82fe03aaa71945a18fbcba569329d8e7/neal_21/
model/82fe03aaa71945a18fbcba569329d8e7/neal_34/
model-audit.json
audit_model.py
examples.xml
shared-selected.xml
preview-gallery.png
preview-neal34.png
```

Model audit SHA256:
`5dc6b4b29541634448591e65a50bad705afd9dc5513fa34b9886f65ebda1132b`.
The owned preview gallery and direct level-34 page rendered without page errors;
the gallery contains 30 entries, of which 22 are Neal reconstructions.

The 630 selected shared action/geometry/feedback/metadata tests passed in 2.18 s.
A broader shared-suite selection stopped at collection because this lightweight
example runtime lacks `requests` needed by unrelated OSWorld/MobileWorld modules;
that attempt is retained as `shared.xml` and is not claimed as a full-core pass.
No dependencies or core modules were changed merely to make that broader check
green.

The final full example run passed **210 tests in 302.64 s**, including all
27 new grid/board/form GUI cases. There are 210 individual JUnit testcase records,
zero failures/errors/skips; the aggregate `tests` attribute says 212 and is not
used as the count. Both this run and the selected 630-test shared run were
independently recounted from individual records. Ruff, all touched JavaScript
syntax checks, documentation links, all 42 private-asset hashes/ignore rules and
`git diff --check` passed. No core-file change, Git staging, commit or push was
performed. The 26 remaining task implementations and the campaign are unfinished.

Example report SHA256:
`be8a315698ea77f4dc3a25fbe7829047e0369e3055b341190f75e04edb75c7eb`.
Selected shared report SHA256:
`528ae8124f2f319884c82bb844e382235e69a20e7cc1b2f186c7dbc48b7e2ea1`.

## Follow-up environment and controller fixes (2026-09-10)

At the time of this validation, the local follow-up was an uncommitted experiment
based on `2ecf495`.
The September 9 results below are retained as the historical baseline, not the
current solvability status. No success condition was relaxed for these reruns.

The first follow-up round used the real `codex exec`, `gpt-6-astra`, `xhigh`, seed 0,
600 seconds and 150 steps for each of ten fresh attempts. **Eight succeeded**:
01, 02, 03, 04, 05, 06, 07 and 10. Tasks 08 and 09 ended as `agent_stopped`, with
complete recordings. These are fixed-instance development attempts, not an
independent model-accuracy estimate.

Changes in that round:

- Tic-tac-toe refresh clears the board and lets X start. The original O-center
  opening and both captured draw sequences remain unchanged. The opponent was
  not weakened and draws still cannot pass. The local candidate is now solvable,
  but its original-site fidelity is still unverified; see the detailed
  [refresh evidence and remaining discrepancy](/examples/not_a_robot/FIRST10.md).
- Generic instructions explicitly permit screenshot-grounded retries. Level 04
  passed after reconsidering its selection, without changing the accepted set.
  This follows the follow-through guidance in
  [OpenAI's Astra documentation](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices).
- The bridge documents lowercase canonical keys and removes the extra 200 ms
  settle sleep between child actions. Every primitive and child screenshot is
  still recorded. Level 10 reached five distinct hits and verified successfully;
  mole spawn/lifetime parameters and the real-time clock were not changed.
- Screenshot acquisition intervals and remaining budgets are returned to the
  model. Follow-up review aligned the budget with the actual controller deadline
  and precisely named the age measurement point as feedback construction.

The first follow-up audit verified all ten archives: **118 model tool calls,
4,417 events and 710 per-attempt PNG files**. It checked event/image hashes,
ordered timestamps, model/effort/workspace evidence, source assets, actual MCP
tool names, model-visible images and decision links. Cleanup was recorded complete
for all attempts. Old port numbers are not used as proof of process ownership:
another service can legitimately reuse a released port.

Artifacts under the operator's `neal-env-fixes.LL1u91` directory:

```text
astra-a/408410254a724393b612b8fb772b1649/
astra-b/16d1b20549c141ee8187cd46b1af423f/
phase1-source.patch
phase1-audit.json
```

Audit SHA256:
`4749b0ed74cf88d1d4dbdfb3c48a496039262270d69875ca370ee507a80f0bf7`.

A separate owned headless-browser probe confirmed that three canonical `ctrl`+`=`
shortcuts leave device pixel ratio, visual viewport scale and game width unchanged.
The bridge now supports a model-selected screenshot crop through `get_observation`.
It returns the original full image plus an enlarged view of those same pixels,
without DOM access, hidden state or changing action coordinates. Crop provenance,
acquisition intervals and decision links are retained. Tasks 08 and 09 were
retested in separate fresh attempts, now both finalized and audited:

| Task | Crop-enabled outcome | Evidence |
| --- | --- | --- |
| 08 License plate | Agent stopped | 20 model tool calls, 11 rejected submissions; no rejected GUI calls |
| 09 Recursive stop sign | Success | 31 model tool calls, 8 rejected submissions before selecting the exact 31-cell reference set |

The accepted inputs and selection masks were unchanged. Across the two follow-up
rounds, **nine distinct tasks have a successful model trajectory**. This is not
a 9/10 single-run accuracy score: round one succeeded on eight of ten attempts;
round two succeeded on one of two targeted retries. All non-successes are retained.

The second audit checked **51 model tool calls, 2,244 events and 371 per-attempt
PNG files**, including each crop's pixel derivation from its archived full image,
source hash, acquisition interval and next-decision link. It also checked the
actual two-image MCP responses against those archives. Both attempts are evaluated,
recording-complete and cleanup-complete; only task 09 has a successful outcome.

Additional artifacts under the same operator directory:

```text
astra-zoom08/18237ffa8b8a4b8d837e4487950931d6/neal_08/
astra-zoom09/a864db5dd0044225af386f06f089dc70/neal_09/
phase2-source.patch
phase2-audit.json
regression-03.xml
shared-01.xml
```

Second audit SHA256:
`9fc502136ba29fb576cb89d158e1643140e73380770bdcfee76e18de47efa2a0`.

Final follow-up engineering verification: **178 example tests passed** in 176.96
seconds, and **630 selected shared tests passed** in 2.04 seconds. The example
count is the 178 individual JUnit testcase records; the aggregate JUnit `tests`
attribute reports 180 and is not used as the count. Neither file contains failed,
errored or skipped testcase records. These test counts are not game-win counts.

### Remaining original-site checks

Task 08's actual keyboard trace contains `867V309`, `867 V309`, `867v309`,
`867U309`, `867V3O9`, `867Y309`, `867 V 309`, then `867V309` again. It never
typed the captured accepted input `867V 309`. This does not demonstrate lost
keyboard input or rejection of the exact accepted string. The supplied
`attempt_101/outcome.json` explicitly leaves whitespace normalization untested.
The original and model-visible screenshots show the same car, plate characters,
prompt and controls; this visual inspection does not establish pixel-perfect parity
or whether the original requires an internal space.

Before changing that matcher, obtain a normal-UI check on an authorized original
game session: capture the pictured plate and try its correct characters without
spaces first. If rejected, capture that result, then submit the spaced reading
on the **same unchanged instance** and capture the completion. If the unspaced
input succeeds, record the submitted value and completion directly. Keep the
screenshots, ordered actions and existing validated official completion record;
no source extraction, cookies or access-control workarounds are needed. A different
plate instance establishes only that instance's behavior, not universal matching.

Task 06 also still needs an original normal-UI refresh-to-win recording. Its local
refresh now permits legal X wins, but the inferred opponent differs from the
center-first winning advice in the public play report. Do not claim original
opponent parity or change its policy merely to obtain a model pass.

A published-demo lead is now identified: the September 8
[GRYOnline report](https://www.gry-online.pl/newsroom/gpt-6-astra-przeszlo-wszystkie-48-poziomow-captcha-ostatnia-obron/z031ad7)
links to [Sharif Shameem's original post](https://x.com/sharifshameem/status/2096847916837314853).
The original post returned HTTP 403 during the read-only check; its gameplay
frames and task-specific inputs were not retrieved. The report is a discovery
pointer, not evidence for changing the opponent or text matcher. A normally
accessible copy of the relevant gameplay segments is still needed.

Neither the full original rules nor all ten model successes were verified in
this follow-up. At that time, its changes had not been committed or pushed; the
remote branch still contained the earlier September 9 implementation.

## September 9 baseline

Validation date: 2026-09-09 UTC. Local branch: `experiment/neal-first10-local`.
Base: `65a61c4` (`Initial public CUA-Lite release`). Only the example directory
is changed. No remote website, access control, training job or GPU was used.

This is a reference-instance experiment, **not ten fully recovered original
levels or a model-accuracy benchmark**. At this baseline, level six was a partial
reconstruction with no reachable win. See the current
[setup and fidelity limits](/examples/not_a_robot/FIRST10.md).

### Engineering verification

- Example suite: **150 passed**, 267.51 seconds. Includes 22 first-ten browser
  cases, 19 MCP bridge cases, 27 Codex launcher cases, plus the existing local,
  fixture, evaluator and recorder regressions.
- Selected shared action/feedback/metadata suite: **630 passed**, 11.90 seconds.
- Ruff checks and Node syntax checks for both game scripts passed.
- Independent read-only reference review verified all 15 imported asset hashes,
  captured accepted inputs, rotation orientation, recursive selection geometry
  and original mole sprite frames. The eight earlier exercises remain separate.

The 22 first-ten browser cases cover ten independent resets, nine known successful
reference-input paths, both recorded level-six draws without false success, and
the recursive terminal-depth selection toggle. Scripted test oracles use visible
DOM geometry and known reference inputs; they are **not model gameplay results**.

The shared selection was:

```text
tests/core/tools/action_space/
tests/gym/utils/feedback/test_action_results.py
tests/gym/utils/feedback/test_feedback_errors.py
tests/gym/utils/feedback/test_extra_tools_surface.py
tests/gym/utils/backend/test_coordinate.py
tests/core/test_metadata.py
```

These are targeted suites, not a claim that the entire repository test suite
passes in the lightweight browser runtime. The container/env-server and canonical
training-data export paths were not exercised.

### Actual Codex attempts

Every final attempt uses the real `codex exec` CLI, `gpt-6-astra`, `xhigh`, a
fresh solver workspace/browser, seed 0, a 600-second budget and a 150-step budget.
The prompt is identical across tasks and contains no answers. The model receives
screenshots and public feedback; only `get_observation`, `computer`, and `finish`
are enabled. Selected model/effort are checked against each matching CLI thread's
turn context. This is client-configuration evidence, not provider attestation.

| Task | Final attempt outcome | Interpretation |
| --- | --- | --- |
| 01 Checkbox | Success | Checkbox completed |
| 02 Stop sign | Success | Captured accepted selection verified |
| 03 Wiggles | Success | Corrected an initial rejected input |
| 04 Vegetables | Agent stopped | Two selections rejected; retained as a non-success |
| 05 Intersection | Success | Visually reassembled and verified |
| 06 Tic tac toe | Agent stopped | Not scorable: current legal state machine has no player win |
| 07 Word search | Success | Both words selected and verified |
| 08 License plate | Agent stopped | Read `867V309`; local captured answer requires `867V 309` |
| 09 Recursive stop sign | Agent stopped | All 31 reference cells plus one extra boundary cell selected |
| 10 Moles | Agent stopped | Two distinct hits; 21 misses before stopping |

Level four's accepted set is the captured carrots/onions/corn combination, not a
general botanical classifier. The final attempt first selected culinary vegetables,
then carrots/onions alone, and stopped. An earlier debugging run reached success
but had an incomplete archive; it does not replace the final non-success.

Level eight is **not an OCR failure**: the model read the plate characters but
omitted the space in the captured successful submission. Whether the original
game accepts an unspaced answer is unverified. This normalization gap prevents
interpreting its local rejection as original-game failure.

Level nine included every captured target cell and additionally selected zero-based
row 5, column 6 in the 16-by-16 terminal grid. Verification rejected this 32-cell
selection. Original boundary tolerance beyond the captured successful set is
unverified; this is a local exact-instance result, not a general segmentation score.

Level ten ran for 188 seconds including controller/environment lifecycle. Moles
continued moving while Astra inferred actions. Two hits and 21 misses were
recorded before the model stopped. The authored timing was not slowed to obtain
a pass, and this does not establish performance on the original site's unmeasured
animation timings.

For level six, independent exhaustive enumeration of the implemented opponent
found zero X-winning paths, 80 O wins and 14 draws. The supplied captures contain
no winning completion. Resolving the original winning interaction requires more
reference evidence; neither the opponent nor the success condition was weakened.

### Artifact index and audit

Raw captures, private artwork and model outputs are intentionally outside Git.
Under the operator's artifact root, the final run directories are:

```text
astra-fixed01-05/bbdbb402fcf44d37bc06b75e33b6a505/
astra-fixed06-10/2a7fc5116efe4c8fb618499d4ec84f09/
```

Each run contains `run_manifest.json`; each task contains exact CLI arguments,
JSONL output, stderr, selected-client-model evidence, final text and `result.json`.
Each trajectory directory contains append-only events, content-addressed PNGs,
the final manifest and a bridge result. Public action summaries are preserved;
private chain-of-thought is neither requested nor exported.

The final read-only audit passed for **all ten complete attempts**: 94 model tool
calls, 2,216 recorded events and 294 per-attempt content-addressed PNG files.
It checked event/image hashes and sizes, ordered
timestamps/sequences, source asset hashes against the submitted files, model/effort
and workspace identity, all actual CLI tool names, every model-visible MCP image
against its archived response, decision-to-observation links and released listeners.
Two model requests used unsupported uppercase keyboard tokens; their failed-tool
responses and subsequent actions were fully retained. A recorded action rejection
is not an archive-integrity failure. The audit output is `final-audit.json` beside
the two final run directory groups.
Outcome, recording completeness and cleanup completeness remain separate fields.
An orderly failed attempt can have a complete archive; that does not make it a
successful game attempt.

Earlier debugging runs are retained separately. One initial run was blocked by
MCP tool approval configuration before browser interaction. Later runs exposed a
real finalization race: the CLI could terminate its MCP child after `finish`
returned but before the archive was published. The bridge now closes resources
and finalizes before replying, with idempotent, cancellation-shielded cleanup.
Those incomplete runs are not repaired after the fact or counted as full passes.

### Remaining acceptance work at the baseline

The immediate missing evidence is a genuine normal-UI level-six win with its
official completion message. Additional original-site evidence is also needed for
text normalization, real-time animation parameters, refresh variants and rejected
submissions. This branch does not reproduce the sequential 1–10 campaign or levels
11–48, and artwork redistribution permission remains unknown.
