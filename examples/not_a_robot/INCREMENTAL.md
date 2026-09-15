# Incremental reference refinement (0.6.0)

This is a scoped refinement of the existing local synthetic game, not an all-48
implementation or original-site benchmark. The catalog still implements 25
Neal task paths plus eight separate authored exercises. No core/Slime behavior,
global credentials or remote branch was changed for this increment.

## Source and what it establishes

User-supplied archive:
`/mnt/weka/home/lingjie.chen/.codex/attachments/fc7567ca-f582-48c1-b46d-b32f98f079c2/neal_all48_incremental_20260910_01.zip`

SHA256: `a887ec7576b86aa933af7818788192f6193f65e99c46bb70d0e6dba0a0981980`.
Size: 1,734,213 bytes. The 123 files include the manifest itself; all 122 listed
members passed size/hash/CRC checks and all 59 JPEGs decoded. No unsafe ZIP paths,
duplicate names, symlinks or encrypted entries were found. Bundled scripts were
not executed; the importer copies only one hash-pinned initial plate screenshot.

Ten attempts contain four visible successes (01/603, 06/602, 07/601, 08/602),
three unknown results and three blocked parent-frame calibration attempts.
There are zero saved official completion messages and no continuous audio/video.
These are DOM-assisted/mixed implementation references, not pure-vision scores.
Among 239 events, 49 action dispatches have 48 returns: level-seven E25 remains
unpaired. Recovery events use UTC-derived elapsed time, not continuous monotonic
capture. These limitations are retained rather than repaired into invented data.

| Level | Used now | Still unknown |
| --- | --- | --- |
| 01 | Additional visible green-check evidence, correctly marked DOM-assisted | Exact loading duration; blocked parent captures are not gameplay failures |
| 06 | Empty-board/X-first refresh and a visible win followed by level-seven transition | Full original opponent, tactical priority/tie distribution, reply timing, official message |
| 07 | Explicit second 10×10 grid; STOPSIGN and BIKE share one selected cell, 11 unique cells | Other placements/generator; missing action return and capture clock continuity |
| 08 | Separate JHB007 photo/input instance with visible transition to nine | Spaced/case normalization; standalone ambiguous attempt is not a success |
| 36 | More legible analysis of the earlier 25 swaps, 0→1260 score and 30→6 moves | No new run; full match/cascade/refill scoring and exhausted-move behavior |
| 41 | Useful close-up brushing, return, flame and bouquet references only | Verify remained disabled; cleaning threshold, accepted sequence and completion |
| 47/48 | Acquisition limitations documented | 47 download: HTTP 403, zero bytes; 48 not downloaded; actual media still absent |

No level-41 page is added from this partial flow. Level 36's swap 14 leaves score
and moves unchanged; the final 900→1260 change is not a universal scorer formula.
The incremental package is helpful but does not remove the remaining 23
implementation gaps. Full priorities remain in [ALL48_PLAN.md](/examples/not_a_robot/ALL48_PLAN.md).

## Explicit instance selection

Import privately with the existing allowlisted importer:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets /path/to/neal_all48_incremental_20260910_01.zip
```

This imports `level08_incremental_reference.jpg` from attempt 602's **initial**
observation, not the filled answer screenshot. Live HTML supplies the input and
Submit; CSS shows only the captured car photograph. Artwork is not Git-tracked
or relicensed. Original assets and existing different user bytes are preserved.

Use `/?task=neal_07&instance=incremental&seed=0` or the equivalent `neal_08` URL
on an owned local preview, or `gym.make("visual_tasks@neal_07",
reference_instance="incremental", seed=0, ...)`. For real screenshot-only attempts:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --tasks neal_07 neal_08 \
  --reference-instance incremental --seed 0 \
  --artifact-root /path/to/a-new-run-directory \
  --browser-executable /path/to/chrome --max-seconds 600 --max-steps 150
```

Only 07 and 08 accept `incremental`. Unspecified instance means `default`; the
campaign keeps default instances and rejects nondefault selection. The seed
controls local dynamics, independently of reference selection. Both the URL and
recorded state/metadata carry the selected instance and its correct source.

The default 07 grid and 12 accepted cells are unchanged. The incremental grid
requires 11 unique cells, with the shared I clicked once. Default 08 still
requires `867V 309`; incremental 08 requires the captured `JHB007`. Each rejects
the other's answer. Exact matching is a conservative local rule, not proof that
the original rejects every other whitespace/case spelling. No normalization was
added to erase an earlier model failure.

## Level-six opponent refinement

The same seeded policy runs before/after a normal refresh: choose among O's
immediate wins, otherwise blocks against X, otherwise all empty cells. This
admits the newly observed edge opening and fork tie. It replaces corner-first
inference that could not produce those observations; it does not hardcode the
recorded winning sequence. Refresh cancels pending moves and continues the RNG;
environment reset restarts the seed. Only a real X line followed by Verify wins.

Regression tests distinguish candidate compatibility with captured replies,
actual GUI gameplay, and genuine screenshot-only Codex attempts. The test-only
minimax player reads visible marks and uses normal input; it is not model evidence
and is never supplied to the model. The older 22/25 score and failed default
plate attempts remain untouched. Current checks and fresh model outcomes are in
[VALIDATION.md](/examples/not_a_robot/VALIDATION.md).
