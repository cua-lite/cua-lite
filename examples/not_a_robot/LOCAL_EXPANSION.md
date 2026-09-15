# Source-derived local expansion

Version 0.7.0 adds 21 playable mechanic variants to the 25 captured-instance
paths. This is **46 local Neal entries, not a faithful implementation of all 48**.
The eight earlier authored exercises are separate.

## Coverage and deliberate differences

All new entries are explicitly marked `fidelity="local_variant"`. The catalog
and trajectory provenance preserve per-level limits and do not label local test
success as an official completion recording. The pages also show a local-variant
notice. None of the old source attempts or historical model scores are replaced.

| Level | Working interaction | Main fidelity limit |
| --- | --- | --- |
| 15 | Continuous steering, collisions, two parking stages | Authored car/scene; no original sound or collision smoke |
| 16 | Animated 3D-looking glyphs, orbit and exact entry | Layered CSS glyphs, not original font meshes |
| 17 | Continuous circle drawing and angular score | Polyline display instead of original drawing effects |
| 19 | Pointer flashlight and five-character entry | Authored dark wall; no original horror video/music |
| 20 | Inkblot image and source-derived text-length acceptance | Only first captured image; nine-image refresh cycle incomplete |
| 23 | Panorama dragging, zoom and camera alignment | Authored spherical illustration, not original photo tiles |
| 25 | Drawing tools, colors and source counter-based acceptance | No semantic drawing recognition; local textures/icons |
| 26 | Two-stage parallel parking with live moving obstacles | Authored car/scene and controls; no original sound |
| 27 | Six colored paths and whole-board coverage | Local CSS drawing; no original click sound |
| 28 | Live prices, source-derived plot/transaction markers, portfolio value and weighted trading cost | Local card/account layout and single-sample drawing; no original audio |
| 32 | Three-round sequence memory, synthesized pad tones | Explicit sound-start gesture and different WebAudio timbre |
| 35 | Three rounds of cup tracking and animated swaps | Authored cups/ball and animation; no original sound |
| 36 | Source-derived generation/refill order, adjacent swaps, cascades, score and move limit | Authored candy shapes; input/animation and asynchronous lifecycle behavior differ |
| 37 | Nine imported face images and source game-label tolerance | Game labels are not independently verified authenticity claims |
| 38 | Continuous driving, people collision and parking test | Schematic scene and substepped local physics |
| 40 | Moving reels, character locking/deletion and submission | Local Unicode fruit, font and background |
| 41 | Gather flowers, brush grave, place bouquet, light candle | Authored scene; no original character animation/sound |
| 43 | View orbit, part dragging and projected attachment snaps | Authored SVG perspective geometry, not original meshes/textures |
| 44 | Legal chess and real Stockfish 17 Lite WASM opponent, Skill Level 10 | chess.js 1.4.0 differs from the original rule library; original deployed WASM byte identity unverified |
| 47 | Original 336-note chart, measured hit timing, played/missed feedback and held-key highlights | Local 102-second synthesized media endpoint; original song/video unavailable |
| 48 | Pause/resume, timed celebration and natural-ending gate | Authored 70-second Canvas/audio finale; no original video/speech |

The complete per-level limitations live in
[tasks.json](/examples/not_a_robot/local/tasks.json). Shared registration remains
in the example, not core CUA-Lite. Rendering dispatch is independent of fidelity
classification; a variant is not mislabeled as a captured instance to make it run.

## Source-rule refinements

Catalog 0.7.1 applies the reviewed plate predicate only to level 08: remove ASCII
spaces and hyphens, then compare case-sensitively. Catalog 0.7.2 refines level 11
using source module 1126: both required Waldo cells must be selected, with at most
one additional cell. The captured three-cell success remains a valid example,
not the only accepted set. Later source-rule refinements below distinguish
other error-tolerant grids from those that require an exact set.
Rule provenance is separate from the original capture records; neither change
adds original-site boundary trials or rewrites earlier outcomes.

Catalog 0.7.3 imports the literal level-47 chart from pinned source module 1070.
The 336 notes retain source order, including its non-monotonic time segment.
Hit timing uses each note's actual spawn time and measured target alignment,
not a backdated nominal spawn. Refresh cancels previous note timers, a deliberate
local lifecycle difference. The local 102-second media endpoint is not evidence
of the original video's natural end or audiovisual synchronization.

Catalog 0.7.4 uses a pinned official Stockfish 17 Lite JS/WASM pair for level 44,
with the source-derived Skill Level 10 and `go movetime 1000` policy. No heuristic
opponent remains. Refresh drains an old in-flight result before accepting moves;
dependency and engine errors cannot award a game outcome. The deployed original
WASM byte identity is unverified, and chess.js 1.4.0 differs in en passant FEN and
repetition behavior. This remains a local variant, not an original-parity claim.

Catalog 0.7.5 gives local runtime failures an explicit `infra_error` state.
Evaluation truncates these attempts instead of recording a game loss or voluntary
model stop. A bounded final state read catches asynchronous failures after the
last observation; an already settled game result is preserved. This changes
failure accounting, not the chess opponent, rules or assistance conditions.

Catalog 0.7.6 restores level 11's source-derived responsive board geometry
(modules 1126, 2025 and 382): desktop width is the parent width minus 50 pixels,
capped at 1800 pixels, with full parent width at viewports up to 800 pixels.
The cell border is chosen at mount as 0.2 pixels below 800 and 1 pixel otherwise.
The 25-by-25 board may require real scrolling to reach Verify. Its image bytes
and acceptance predicate are unchanged; shared card chrome and checkmark artwork
remain authored approximations.

Catalog 0.7.7 aligns the shared level-15/26 parking renderer with source module
511's `loadLevel`, `resetGame` and wheel drawing: manual refresh, collision reset
and internal stage changes preserve held inputs. Two visible front wheels use
the source geometry and current steering angle, including stationary steering.
Blur and shutdown still release input. Vehicle motion, collision detection and
the parking predicate are unchanged. The body, scene, controls and missing
audio/smoke remain explicit local differences.

Catalog 0.7.8 aligns level 36 with source module 1078's board-generation procedure:
generate all 64 candies, then repeatedly redraw complete matching sets, scanning
all rows before all columns. Refill consumes new draws from the lowest empty row
upward within each column. The local seeded stream is not the original site's
random stream. Score, move limits and Verify are unchanged; in-flight refresh
still cancels old work, and the catalog retains input/animation/lifecycle limits.

The campaign controller also classifies an asynchronous dependency failure seen
in a new stage's first observation as an immediate infrastructure truncation.
The previous stage's completion remains archived; it does not award campaign
success or keep the errored stage active.

Catalog 0.7.9 aligns level 06 with source module 1121: a 100 ms opening, 450 ms
replies, source-ordered tactical decisions and the 40% random branch after
refresh. The first game's chance draws still advance the seeded stream, even
though only later games may use the random branch. Refresh cancels old timers,
an intentional lifecycle difference. The original marks/strike animations and
audio remain unavailable; historical capture and model results are unchanged.

Catalog 0.7.10 recovers two more first-ten rules. Level 04 uses module 1125's
required/optional image labels and combined one-error tolerance, rather than
accepting just the captured selection. Level 10 follows module 1127's immediate
spawn, score-dependent lifetimes, random spawn intervals and mutable hit set.
Its literal 1–16 spawn pool is retained despite module 379's 0–15 rendered grid.
At the hit threshold only the latest hide and next-spawn timers are cancelled;
remaining active moles and older hide callbacks retain source behavior. Refresh
and shutdown still cancel every owned timer as an explicit lifecycle difference.
Old attempts are not regraded, and neither change restores original audio or
proves a new original-site boundary trial.

Catalog 0.7.11 restores level 28's visible chart feedback from source modules
1119 and 2009. The 30-sample history retains per-sample buy/sell counts, including
same-sample overlap and source drawing order. The plot spans the current samples,
uses the source 0.97/1.03 range padding, 20/40-dollar axis spacing, and current
price marker. Its backing canvas is twice the card width and 440 pixels high,
displayed at 220 pixels. The portfolio value is shown separately from realized
profit; the 500 ms price process, weighted cost and 2,500-dollar predicate are
unchanged. One sample is explicitly drawn at the right edge without an invented
history fill, instead of the source's nonfinite coordinate. Refresh clears the
plot and keeps buying disabled until the next tick. The card/account layout
remains local, original audio is absent, and this source alignment is not an
original-site replay or model success.

Catalog 0.7.12 restores level 47's note and key feedback from source modules
1070 and 1867. A hit pauses the falling animation, fades/scales for 200 ms and
removes the note 300 ms after the hit. A miss dims/scales without pausing; the
native falling-animation end schedules removal 300 ms later. Ending a round
stops judgment while existing note feedback continues. Held arrow keys show
the source-derived 100 ms pressed transition and final 1.06 scale. Pointer
capture extends the original touch controls to mouse release outside a target.
Refresh and shutdown cancel owned visual timers, remove old nodes and clear
pressed keys; this deliberately differs from the original refresh races.
The chart, hit window, accuracy thresholds and local media endpoint are unchanged.
The card, base note artwork, particles and original audiovisual fidelity remain
explicit differences; this feedback alignment is not a model-success claim.

Catalog 0.7.13 aligns three fixed-image predicates with pinned source modules.
Level 09 (1100/530) accepts at most two combined missing and extra depth-four
leaves; all 31 source paths map to the existing 16-by-16 mask. Level 29 (1117)
uses its three source-positive images, permits one combined error and always
rejects the forbidden fourth image. Its previously captured four-image answer
remains valid, but is not the only answer. Level 46 (1077) requires all eight
floor cells and permits up to two extra cells. Levels 02/12/13/31 retain their
exact predicates. Original asset URLs, import manifests and file hashes were
matched independently of the captured success attempts. Source-derived rules
do not turn those separate asset loads into same-attempt evidence, regrade old
attempts, or establish new original-site boundary completions. Original audio,
card/selection artwork and recursive split animation remain unreproduced.

Catalog 0.7.14 gives level 07 source-derived refresh generation (module 1072).
The two captured initial boards remain unchanged; the visible Refresh control
now generates a new 10-by-10 puzzle and clears selected cells. BIKE is placed
before STOPSIGN, following the source's alphabetic order. Both use the eight
source directions, row-major candidate order and maximum-overlap preference.
Blank cells consume the source's 23-letter alphabet in row-major order. Verify
uses the union of placed word cells, not a scan for incidental visible words.
With these fixed two words, the source's general retry/growth branches cannot
be reached: BIKE occupies at most four rows, leaving empty rows for STOPSIGN.
Randomness remains locally seeded, and initial mounting still uses a captured
board instead of drawing a random one. Original audio and card/grid styling
remain incomplete. This does not add original-site refresh recordings or
rewrite the historical completion evidence of either captured instance.

Catalog 0.7.15 aligns level 05 Refresh with source module 1111: consume one
local random draw per tile in grid order and choose among four quarter-turns.
Clicks still accumulate clockwise turns, and Verify requires every angle to be
a multiple of 360 degrees. Module 1993 supplies the 200 ms ease-out transition.
Initial mounting retains the captured arrangement instead of sampling; local
tile clipping/layout and missing original sound remain explicit differences.

Levels 01 and 14 share module 478's feedback semantics: any point on a checkbox
card starts loading, a correct choice displays a green check after 700 ms and
emits completion after 1600 ms, while a wrong choice retains a red cross after
800 ms without completion. The fixed 56-label list in module 1096 requires no
random generator. Repeated clicks retain independent pending callbacks and do
not clear an existing check/cross. Local terminal handling cancels remaining
callbacks and prevents duplicate completion; reset destroys the owned page.
Checkmark/spinner artwork and missing original sounds remain local differences.
These source-derived rules do not replace historical captures or
claim new original-site timing recordings.

Catalog 0.7.16 restores the official embed geometry for those checkbox cards.
Source module 564 overrides the main site's margins: level 01 stays at the
top-left, at most 315 px wide. Level 14 uses module 1899's responsive 4/3/2/1
columns, with breakpoints at 1400/1060/700 px, 13 px padding, 10 px gaps and a
1500 px maximum width. It scrolls the document rather than a fixed-height inner
panel. Source modules 283/508 supply border-box sizing, 76 px card height and
normally wrapping labels. Original logo/font files and sound bytes are not
bundled; the authored branding, mark/spinner animations and font fallback remain
explicit differences. This is source-derived geometry, not an original-site
pixel-equivalence claim.

Catalog 0.7.17 preserves source module 511's independent physical-key and
on-screen-control ownership for levels 15/26. Releasing one steering or gas
alias no longer releases another held owner. Physics, collision and parking
predicates are unchanged. Fixed local scene sizing, frame timestamp initialization
and cancellation of old collision-reset callbacks remain explicit differences.

Level 40 follows modules 1115/414: input length is compared with the sampled
answer length, not the previous input length. Bulk insertion is one sample, and
an ignored sixth character followed by backspace can append an extra sample.
Unchanged input values do not resample. The local empty-input/empty-answer path
remains a no-op instead of reproducing the original callback error; full Vue
watcher batching, IME timing and refresh lifecycle parity are not claimed.
Regression tests use trusted keyboard/pointer inputs and DOM-derived test
oracles with a virtual clock, not screenshot-only model gameplay or new
original-site completion evidence.

## Not complete

- **42 and 45:** the original remote dialogue judge was not recovered. No canned
  response sequence, unconditional success button, or fabricated provider call
  substitutes for it. These two tasks remain absent from the playable registry.
  A materially different offline dialogue judge needs a separate explicit choice.
- **Media/spatial fidelity:** the original 3D assets, some fonts, sound effects,
  songs and finale are not bundled. The level-47 chart is privately imported,
  not distributed with the code. Authored alternatives above are local
  candidates, not evidence that those original-material gaps were solved.
- **39:** unchanged unavailable-camera path only, not facial-expression detection.
- **Full campaign/model evaluation:** no new 48-stage model run or original-site
  success collection is claimed. A sequential campaign still stops at missing 42.
- **Audio capture:** WebAudio playback exists for 32/47/48, but screenshot-only
  model input does not include sound. Tests do not establish recorded audio
  quality or original audiovisual synchronization.

## Local runtime

Use Python 3.12 with
[requirements.txt](/examples/not_a_robot/requirements.txt), plus an existing
Chromium-family browser or a separately installed Playwright Chromium.
No GPU, Docker, WSL, remote service or model API key is needed for local preview.
Pinned chess.js (BSD-2-Clause) and Stockfish JS/WASM (GPL-3.0) dependencies are
served locally. Level 44 needs a browser with WebAssembly and Worker support,
not an installed native chess engine or a network service. See their
[provenance and hashes](/examples/not_a_robot/local/vendor/spatial/PROVENANCE.md).
The original site's application JavaScript is not vendored.

Import the three reviewed ZIPs and the pinned rhythm analysis module with the
commands in [SUPPORT.md](/examples/not_a_robot/SUPPORT.md). The importer checks
input and selected-output SHA256 values. The allowlist contains 65 images and
one derived chart JSON. It neither executes nor serves the analysis module and
does not copy arbitrary capture logs, cookies, browser profiles or raw traces.
All imported material remains Git-ignored; capture is not a redistribution license.

Example PowerShell preview, from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
uv run --no-project --python 'C:/path/to/python.exe' python -m examples.not_a_robot.local_tasks --port 8765
```

Open `http://127.0.0.1:8765/?task=neal_15&seed=0`. Use the gallery for
independent tasks. Ctrl+C stops only that preview. Evaluation creates and closes
its own loopback server and browser, so no separately running server is needed.

The Windows transport now polls pipes without socket-only `select()`, restores
the caller-owned descriptor and uses Windows signal/process-group handling.
POSIX directory metadata fsync is not invoked on Windows; individual file fsync
and atomic final manifest replacement remain in place.

## Verification interpretation

### Optional continuous video

Pass `record_video=True` to `gym.make`, or add `--record-video` to `smoke`,
`codex_smoke`, or `codex_bridge`. Recording is disabled by default. The owned
browser context saves a silent WebM at the requested viewport size. The main
page's video is finalized before the attempt manifest is published; a campaign
keeps one main-page video across its navigation boundaries.

`manifest.video.status` is `not_requested`, `saved`, `missing`, or `failed`.
`saved` means the file was finalized, found nonempty, and hashed, not decoded or
visually verified. Playback verification remains separate. Video saving adds
neither video nor audio to screenshot-only model input. The native recorder has
no audio track. Event timestamps bracket the lifecycle but do not establish a
precisely synchronized video time zero.

`recording_complete` covers events and PNGs only, as stated by `recording_scope`.
Results expose `video_requested`, `video_saved`, and the video descriptor
separately from the game outcome. Missing video is incomplete video evidence,
not evidence that the game itself failed.

### Ordered screenshot sequences

The Codex bridge's observation protocol 1.1.0 adds an optional `sequence` to
`get_observation` and `computer`. It contains required numeric `duration_seconds`
(greater than zero and at most 10) and `interval_seconds` (0.1 through 10).
It cannot accompany a `region`. The normal call's screenshot is the first frame;
subsequent frames use canonical screenshot steps and consume the episode budget.
Initial observations can request a sequence. Computer actions execute first, then
sampling starts without waiting for another model decision.

Responses contain at most 64 ordered frames and 8 MiB of PNG data. Actual capture
start/end times, frame hashes, target offsets, lag, skipped slots and stop reasons
describe what was collected. A slow capture does not produce backfilled frames.
Terminal, infrastructure, budget and campaign boundaries can end sampling early.
Earlier action errors remain in the response even when later screenshots succeed.

An archived frame is not necessarily returned: for example, a new screenshot may
exceed the remaining byte allowance. Decision provenance refers to the last
prepared response's frames, excluding such omitted captures. `controller_result`
records a prepared response; `controller_response_sent` is emitted only after a
successful transport write and flush while the recorder remains open. Neither
record proves the model understood an image. Finish finalizes the archive before
its response is sent, so its transport delivery requires the separate client log.

The CLI's JSONL event projection can truncate large MCP results into text. That
projection alone does not establish which images entered model context. For a
specific authorized attempt, inspect only its tool-output image records and
compare their hashes and order with the bridge's prepared frames. Do not export
raw session contents or private reasoning as evidence. Client-side image records
still do not prove the provider's model identity or comprehension.

These are multiple still-image inputs, not native video or audio. The optional
silent WebM remains separate recording evidence. More frames consume model input
and environment steps, and the game continues during model inference. Sequence
support does not establish that a model can meet every game's real-time demands.

### GUI and model checks

The new browser suites use actual mouse/keyboard inputs and inspect visible
controls or evaluator events as test oracles. They do not call the success
function directly or overwrite game state. Many animation tests use a browser
test clock, explicitly different from real-time gameplay recordings. Additional
parking/trading checks exercise real wall-clock motion while idle.

Run with a new artifact directory each time:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NEAL_BROWSER_EXECUTABLE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
uv run --no-project --python 'C:/path/to/python.exe' python -m pytest -p no:cacheprovider -o addopts= --basetemp='C:/path/to/new-test-artifacts' -m 'not stress' examples/not_a_robot/tests
```

On Windows without symlink-creation privileges, the two tests requiring real
symlinks are reported as skipped. Their asset-import rejection logic is retained.
No model result follows from a passing GUI-oracle suite. Use a separately
authorized, bounded model evaluation if screenshot-only task performance is needed.
