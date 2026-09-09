# First-ten validation record

Validation date: 2026-09-09 UTC. Local branch: `experiment/neal-first10-local`.
Base: `65a61c4` (`Initial public CUA-Lite release`). Only the example directory
is changed. No remote website, access control, training job or GPU was used.

This is a reference-instance experiment, **not ten fully recovered original
levels or a model-accuracy benchmark**. Level six is still a partial reconstruction
with no reachable win. See [setup and fidelity limits](/examples/not_a_robot/FIRST10.md).

## Engineering verification

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

## Actual Codex attempts

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

## Artifact index and audit

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

## Remaining acceptance work

The immediate missing evidence is a genuine normal-UI level-six win with its
official completion message. Additional original-site evidence is also needed for
text normalization, real-time animation parameters, refresh variants and rejected
submissions. This branch does not reproduce the sequential 1–10 campaign or levels
11–48, and artwork redistribution permission remains unknown.
