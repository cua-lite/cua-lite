# Reference fidelity and evaluation contracts

For installation and launch commands, use the
[README](/examples/not_a_robot/README.md). This document describes the current
local game's evidence boundaries, non-obvious mechanics and recording contracts.
It is not an original-site specification or a record of individual model runs.

## Scope and sources of truth

The catalog contains **46 local Neal task paths**: 25 captured-instance paths
and 21 source-derived local variants. Eight authored exercises are separate.
Levels **42 and 45 are not implemented**; their remote dialogue judge is missing.
Level **39 implements only the observed unavailable-camera branch**, not facial
expression recognition. A registered path is not a claim of complete fidelity.
The sequential controller cannot complete all 48 because it stops at missing 42.

The following files own the detailed data; this document does not duplicate
their asset lists, source hashes or per-attempt provenance:

- [tasks.json](/examples/not_a_robot/local/tasks.json): catalog version, task
  identity, reference instances, rule sources, completion evidence and limitations.
- [reference_manifest.json](/examples/not_a_robot/reference_manifest.json):
  accepted archive hashes, exact imported members and derived-file identities.
- [reference_assets.py](/examples/not_a_robot/reference_assets.py): private import
  validation and the literal-chart derivation, without executing source scripts.
- [local/](/examples/not_a_robot/local): current renderers and page evaluators.
- [env.py](/examples/not_a_robot/env.py),
  [codex_bridge.py](/examples/not_a_robot/codex_bridge.py) and
  [recorder.py](/examples/not_a_robot/recorder.py): lifecycle, observation and
  archive contracts. Per-run manifests describe the version actually evaluated.

`captured_instance` means an evidence-bounded reconstruction of particular
instances. It does not mean pixel identity, complete random-instance coverage or
an original random generator. `local_variant` means source-derived mechanics with
explicit local changes, including substantial visual or media substitutions.
The separate authored adaptations use `mechanic_adaptation`, not either claim.

Keep these forms of evidence distinct:

1. A capture may show a visible terminal state or a normal next-level transition.
2. A saved official completion message is a different, stronger interface record.
   Its recorded origin/source and same-attempt identity must be checked.
3. Reviewed source can establish a rule not tested by the captured input, but
   does not create a new original-site trial or repair an unknown old outcome.
4. A local GUI-oracle test checks the implementation. A screenshot-only model
   attempt checks that model on the specified local version and instance.

Read `completion_basis`, `official_completion_message_recorded` and
`policy_mode` alongside `completion_recorded`. In particular, a visible success
can have `completion_recorded=true` without a saved official message. DOM-assisted
captures, engine-assisted games and observations exposing hidden state are useful
implementation references, not clean screenshot-only benchmark trajectories.
No message, no page change or a clickable Verify button alone proves failure or
success. Source-rule improvements do not regrade earlier attempts.

## Private assets and instance selection

The importer accepts the reviewed `first10`, `expansion` and `incremental`
archives identified by the manifest. The expansion archive's basename refers to
31-48, but its allowlisted members also include 11-30. Duplicate archive trees
and repeated manifest records are not unique gameplay assets.

The allowlist contains 65 images and one derived rhythm chart. Only selected,
hash-verified members are copied. Different existing bytes and symlinks are
refused; arbitrary ZIP contents, scripts, executables and capture logs are not
extracted or served. Separate imports preserve unrelated already imported assets.
Absent required images or a missing chart are boot errors, not silently generated
replacement data. See the README for the runtime's private-asset requirements.

The chart importer accepts the pinned analysis module and extracts its literal
note array into `level47_chart.json`. It verifies the derived hash and preserves
all 336 entries in source order. It neither executes nor serves the module.
The original application JavaScript is not vendored with this example.

Original artwork and the chart remain private and Git-ignored. Access to a
capture does not confer redistribution rights. The importer does not collect
cookies, login tokens, browser profiles, raw HAR files or unrelated website data.
Bundled third-party libraries have separate licenses, described below.

### Why some assets are screenshot crops

Levels 08, 12, 18, 24 and 33 display CSS crops of unchanged reference screenshots
where separate source photos do not establish the same instance. Only the game
image region is displayed; captured inputs, surrounding UI and completion logs
are excluded. Live HTML supplies the interactive controls. JPEG capture quality
is retained, not relabeled as lossless or replaced with invented detail.

- Level 08's two photo instances use their respective initial screenshots, not
  filled-answer screenshots. A different car image cannot inherit an accepted
  string from another attempt.
- Level 18 uses nine same-attempt frames, including stable replacement states.
  The initial parent scroll offset differs, so its crop origins differ. Eight
  separately captured photos do not cover the full replacement sequence.
- Level 24 retains four stages from one attempt. Separate image loads do not
  establish that a new color, blur or text instance shares its accepted inputs.
- Level 10's 400-by-200 sprite has normal and hit frames, both used locally.
  The presence of those frames does not establish original animation or audio.

### Instances, seeds and reset

Only 07 and 08 accept `reference_instance="incremental"`, or `instance=incremental`
in an owned preview URL. Unspecified selection means `default`; unknown pairs are
rejected. The campaign uses default instances and rejects nondefault selection.
Instance selection and `seed` are independent and recorded in task provenance.

Environment reset creates a new episode and restores its selected initial
instance. In-page Refresh preserves episode events and mistake accounting and
follows that task's rules. It is not universally a new original random instance:
05/07 generate new local-seeded arrangements; 06 starts an empty X-first board.
The source-derived random algorithms still use a local stream, not the site's
random stream. Seeds do not reproduce wall-clock input, every animation frame,
font fallback or engine scheduling across browser versions and machines.

## Non-obvious current mechanics

Exact source-module identities and limitations remain in the catalog. The notes
below explain semantics that are not fully represented by those fields. They are
implementation/reference guidance, not additional information supplied to solvers.

### Checkboxes, text and image grids

- **01/14:** the whole checkbox card is clickable, including its logo/background.
  Correct feedback appears at 700 ms, but completion waits until 1600 ms. A wrong
  choice retains its red cross after 800 ms without completing. Repeated clicks
  retain independent callbacks and do not clear an existing mark. Local terminal
  handling cancels remaining callbacks and prevents duplicate completion.
  The fixed 56-label list is not randomized. Embed geometry uses document scroll,
  not a fixed-height inner panel; layout dimensions are in `checkbox_layout`.
- **02/12/13/31:** Verify uses the exact accepted set for each fixed captured
  instance. Other source images, refresh coverage and original-site boundary
  trials remain incomplete; this is not a general segmentation classifier.
- **03/08/33:** do not generalize one text task's normalization to another.
  Level 08 removes only ASCII space and hyphen before case-sensitive comparison;
  nonbreaking spaces and Unicode hyphens remain significant. Its two instances
  remain separate. Level 03 uses exact comparison and authored whole-raster
  deformation; broader normalization and original deformation/audio are unverified.
  Level 33 retains its captured glyph quality and unverified normalization limits.
- **04/09/11/29/46:** a captured successful set is not necessarily the only valid
  set. Their `selection_rule` records required/optional cells and error tolerance.
  Level 04 has a combined missing/wrong budget, not a botanical classifier.
  Level 09 selects depth-four leaves after recursive splits; its error budget
  covers missing and extra leaves together. Level 29 always rejects its forbidden
  image, even within the ordinary error allowance. Its labels are fictional game
  criteria. Level 46 requires every floor cell and permits limited extras.
  Matching asset hashes does not turn a separate asset load into same-attempt
  completion evidence or establish new original-site boundary trials.
- **11:** its 25-by-25 board follows source-responsive sizing and may require real
  scrolling. Width responds at 800 px; the thin/thick cell border is chosen once
  at mount, with the thin border only below 800 px. Card and checkmark art differ.
- **05:** Refresh draws one quarter-turn per tile in grid order. Clicks accumulate
  clockwise turns; Verify checks multiples of 360 degrees. The 200 ms ease-out
  transition is source-derived, but initial mounting uses the captured arrangement
  and local clipping/layout rather than a newly sampled original board.
- **07:** Refresh places the two fixed words alphabetically, using eight source
  directions, row-major candidate order and maximum-overlap preference. Remaining
  cells use the source's 23-letter alphabet in row-major order. Verify checks the
  union of placed cells, not incidental words. With these two words, retry/growth
  is unnecessary: the first occupies at most four rows, leaving clear rows for
  the second. The captured initial boards, including their different overlap,
  are preserved independently of generated Refresh puzzles.

### Turn-based and dynamic interactions

- **06:** O opens after 100 ms; replies to X wait 450 ms. Each opponent turn with
  legal cells consumes a chance draw. After at least one Refresh, a draw below
  0.4 uses a second draw to choose an empty cell. Otherwise source-ordered O wins,
  X blocks, center, corners 0/2/6/8, then the first empty cell determine the move.
  Winning-line priority is row/column/diagonal order, not empty-cell order.
  Refresh increments the game count and continues the stream; reset restarts both.
  A real X line followed by Verify is required. Draws/losses do not succeed.
  Cancelling old timers on Refresh/close is an explicit local lifecycle difference.
- **10:** the first mole appears immediately. Spawn interval, lifetime by current
  hit count and index pools are in `mole_rule`. The source's 1-16 spawn pool and
  rendered 0-15 grid disagree; index 16 remains off-grid and cell zero never spawns.
  Selection and hit state are independent: empty selection earns no hit, clicking
  a hit removes it, and a preselected cell must be toggled off/on to hit a mole.
  At five hits only the latest hide and next-spawn timers are cancelled. Earlier
  hide callbacks and remaining moles persist, allowing another hit. Cancelling a
  hit can disable Verify without restarting spawns. Verify needs at least five
  current hits. Refresh/close cancel all owned timers; pop animation/audio differ.
- **18:** each cell owns its captured photo queue. A target click replaces only
  that cell after 650 ms; pending duplicates are ignored. Distractor selections
  toggle and block Verify. Completion needs no targets, no pending replacement
  and no selected distractor, not a global click count. Queue independence,
  dimming, delay, duplicate suppression and distractor behavior are local policies.
  Refresh restores the same queues and cancels pending work.
- **22:** nine entities independently change from roaming to returning to caught.
  Verify requires all nine caught; reclicking a returning/caught duck cannot
  advance another. Local fixed parameters are a 600-by-590 px field, 100-by-100 px
  input rectangles, seeded 100-160 px/s reflected paths, 160 ms sprite frames and
  300 ms returns. The net is 100-by-66.7 px with a 27/20 px hotspot. These are
  inferred, not measured original timings or transparent-pixel hitboxes. Ordinary
  current browser hit testing applies, without snapping or matching an old frame.
  Missed clicks are recorded separately from rejected-Verify mistake counts.
- **15/26:** physical aliases and on-screen controls are independently held and
  OR-combined. Refresh, collision reset and internal stage changes preserve held
  input; blur/shutdown release it. Visible front wheels reflect steering even
  while stationary. Fixed local scene sizing, artwork, timer cancellation and
  frame initialization remain differences. Level 38 separately uses substepped
  local physics and source-derived corner collision, not full polygon collision.
- **28:** trades buy/sell one share. Inventory uses weighted-average cost; profit
  is realized on sales, while portfolio value is current price times holdings.
  Verify accepts cash or realized profit of at least $2,500, not portfolio value
  alone. The real-time 500 ms price process is not a replay advanced by clicks.
  Its 30-sample plot retains per-sample buy/sell markers, current-price feedback,
  source range padding and axis spacing. Same-sample trades overlap in source
  drawing order. A lone sample stays at the right edge without invented history
  fill. Refresh clears the ledger/history and disables buying until the next tick;
  the card/account layout and this single-sample case remain local differences.
- **36:** generate all 64 candies, then redraw complete matching sets, scanning
  rows before columns. Refill draws bottom-up within each column. Cascades and
  score are actual state transitions, not a replay of captured swaps. A victory
  overlay is not the task's Verify submission. Seeded randomness, pointer-up
  swapping, animation, immediate post-cascade game-over and cancellation of old
  work remain local differences; no automatic dead-board reshuffle is added.
- **40:** input length is compared with sampled-answer length, not prior input
  length. Bulk insertion produces one sample; ignoring a sixth character and
  returning to five can append another. Deletion only releases the old sampled
  suffix; it does not necessarily unlock every reel after the visible input.
  Unchanged values do not resample. The empty-input/empty-answer case is a local
  no-op; original callback errors, full Vue watcher batching, IME timing and
  Refresh lifecycle are not reproduced.

### Other local variants and incomplete capabilities

The remaining renderers use actual task state rather than arbitrary click counts:
inventory recipes consume materials (21), slides require the adjacent empty cell
and solved arrangement (30), and math ordering checks numerical order (34).
Extra inventory merge/swap semantics and several captured-task reset behaviors
remain inferred. Drawing and spatial variants have more substantial limits:

- 16 uses layered glyphs rather than original font meshes; 19 uses an authored
  dark wall rather than original horror media; 23 uses an authored panorama.
- 17 scores a polyline circle. Level 25 follows source counters, not semantic
  drawing recognition. Level 27 requires colored paths and board coverage.
- 32 uses an explicit sound-start gesture and local WebAudio timbre. Level 35's
  three-round tracking uses authored cup/reveal art; Refresh preserves positions.
- 37's fixed image labels are game criteria, not independently verified claims
  about whether a person's picture is AI-generated. Level 20 has only one of the
  original nine inkblots, so its image-refresh cycle is incomplete.
- 41 has authored grave/flower/candle interactions. Level 43 uses projected SVG
  part snapping and visible attachment dots, not original meshes or strict 3D
  topology. Full per-level differences remain in `reference.limitations`.

**39:** the reference showed a camera prompt and `0% HAPPY`, yet Verify produced
a completion record under unavailable-device conditions. The local branch
reproduces that limited result as `reference_camera_unavailable_path_completed`.
It requests no device, stream, permission, face image or synthetic face.
`camera_flow_tested`, `facial_expression_detection_verified` and
`human_identity_verified` are false. A broader camera workflow requires separate
informed authorization and retention rules; this branch does not supply them.

**42/45:** an observed conversation is not an arbitrary-response judge. No canned
dialogue, unconditional success or fabricated provider response substitutes for
the missing original backend. A different local/API judge would be a separately
authorized, explicitly labeled variant, not recovery of the original rules.

**44:** the opponent is real Stockfish 17 Lite in a same-origin Worker, at Skill
Level 10 with `go movetime 1000`; there is no heuristic fallback. Refresh drains
an old search result before accepting moves, and close terminates the owned
Worker. Dependency/protocol failures cannot award a game outcome. A genuine White
checkmate remains verifiable without a further engine reply. chess.js 1.4.0
differs from the unpinned original rule library, including en passant FEN and
repetition behavior. The loader matches captured bytes; the original deployed
WASM identity and move-for-move scheduling parity are unverified.

The bundled chess.js BSD-2-Clause and Stockfish GPL-3.0 notices, pinned versions
and corresponding-source links are in
[vendor provenance](/examples/not_a_robot/local/vendor/spatial/PROVENANCE.md).
Retain those notices and applicable source-distribution obligations. Their
licenses do not license Neal artwork or audio. Level 44 needs Worker/WebAssembly
support, not a native engine installation or an external engine service.

**47:** the imported 336-note chart retains its non-monotonic segment. Judging
uses actual spawn time and measured target alignment, not backdated nominal time.
The earliest unjudged spawned note in a lane is judged once, with strict timing
error below 0.25 seconds. After more than 20 judgments, accuracy below 80% ends
the round early. The local media endpoint requires at least 85%, then Verify.
A hit pauses/fades/scales and is removed after 300 ms; misses continue falling
and remove after animation end plus 300 ms. Ending judgment leaves existing
feedback visible. Held-key feedback is separate from scoring. Refresh/close
cancel owned work; mouse pointer capture extends the original touch controls.
The 102-second synthesized media endpoint is not the original video ending.

**48:** the authored 70-second Canvas/audio finale supports pause/resume and a
natural-ending gate. It is not the original video, speech, music or certificate.
Direct completion of this independent level does not prove prior levels were
played. WebAudio playback in 32/47/48 is separate from screenshot-only model input.
Original audio quality and audiovisual synchronization remain unverified.

## Sequential campaign contract

`visual_tasks_campaign@full_game` starts at 01 with the fixed 48-stage order.
It is distinct from independent `visual_tasks@neal_NN` and the remote prototype
`not_a_robot_campaign@full_game`. The preview gallery is independent navigation,
not another campaign controller. Only the environment owns stage transitions.

- A page evaluator must report success. Before navigation, the environment saves
  that page's final screenshot, state and remaining events, then advances one
  position. Timer completion can be recognized by an observation-only call.
- Browser, context, server, resource identity, recorder, steps and elapsed budget
  persist. Each stage receives the same local seed, not an original-site seed.
- A completed stage stops the current action batch. Its unexecuted tail is
  reported and is not replayed on the new page. The next decision needs the new
  screenshot. Owned held keys/buttons are released on the old page and logged as
  lifecycle input, not attributed to a fresh model decision.
- A missing next task produces `unsupported_task`, truncation and reward 0.
  Successful displayed-page state does not imply campaign success. The launcher
  marks `blocked_by_missing_task=true` and excludes the attempt from model grading.
- Budget exhaustion truncates; task failure terminates. Navigation/observation
  or new-stage dependency failures remain infrastructure errors, preserving the
  completed prefix. Only ordered completion of all 48 earns campaign reward 1.
- `finish` stops a partial campaign as `agent_stopped`, even if its current page
  just completed. It does not navigate to the next stage or award full-game
  success. Reset finalizes the old attempt and starts a new campaign at 01.

Global event sequence and episode elapsed time continue across navigation;
page-local event sequence/time restart. `campaign_stage_started` records reference
metadata, `campaign_stage_completed` links terminal state and screenshot,
`campaign_boundary` preserves the exact unexecuted tail, and `campaign_end`
records the prefix/outcome. An injected missing-stage test or controller fixture
does not establish gameplay through the corresponding real stages.

## Input and observation boundaries

The local model controller exposes only `get_observation`, `computer` and
`finish`: screenshots, visible instructions and public progress/budget/errors.
It does not expose selectors, DOM/source, evaluation targets or hidden state.
Canonical coordinates are normalized to 0-1000 in the full screenshot, not
pixels. Keyboard tokens use canonical lowercase names such as `["ctrl", "a"]`.
The environment separately records requested actions and executed GUI primitives.

Held keys survive observations and no-tool intervals until released; `key_down`
does not promise OS auto-repeat. Canonical drag uses a sequence of real pointer
moves, not instantaneous placement. Native text/image dragging can affect pointer
delivery, so task surfaces and calibration matter. There is no additional bridge
post-action settle delay, but input, screenshots and recording still take time.
Real-time games continue during inference and observation; a recent screenshot
does not guarantee the same state when the next action arrives.

### Region and ordered screenshot sequences

`get_observation(region=[left, top, right, bottom])` returns the full image plus a
nearest-neighbor enlargement of the same pixels, up to 4x with a 1600 px maximum
edge. It neither changes the game scale nor adds detail. Actions still use full
image coordinates. The crop's source hash, pixel rectangle and acquisition
interval are archived. Browser zoom shortcuts are not assumed to change headless
page scale.

Protocol 1.1.0 supports `sequence` on `get_observation` and `computer`, with
`duration_seconds` in `(0, 10]` and `interval_seconds` in `[0.1, 10]`, both finite
numbers. It cannot accompany `region`. The normal call result supplies the first
frame; further frames are ordinary screenshot steps that consume the same budget.
The first observation may request it; a computer action batch executes before
sampling. No additional gameplay input is sent during sampling.

A response contains at most 64 ordered frames and 8 MiB of PNG bytes, including
the first frame. Slow capture or delayed scheduling skips expired slots rather
than reconstructing them. Check actual acquisition intervals, target offsets,
lag, skipped slots and stop reason, not nominal spacing alone. Terminal, tool,
infrastructure, budget and campaign boundaries can end a sequence early. Earlier
action errors are not hidden by later successful screenshots. These are still
images, not native video/audio; more frames consume model input and episode steps.

### Prepared, sent and actually returned images

Archive-only frames, including an image exceeding the response byte cap, must
not be counted as model input. `controller_result.model_visible_frames` identifies
prepared frames; the next `model_decision` links to that prepared observation,
crop or sequence. `controller_response_sent` is emitted only after transport
write and flush succeed while the archive remains open. Neither proves model
understanding or provider identity.

`finish` closes resources and finalizes the archive before replying. Repeated
finish/close cannot append to the finalized episode, so finish-response delivery
requires separate client evidence. CLI JSONL projections can truncate large MCP
outputs into text. For an authorized delivery audit, compare actual tool-output
image hashes/order with prepared frames; projection truncation alone proves
neither delivery loss nor complete receipt. Keep an explicit unknown when the
permitted records cannot settle it. Do not export raw sessions/private reasoning.

## Archives, video and evaluation

Each reset creates a new attempt directory. Events are ordered JSONL; PNGs are
content-addressed and carry size/hash descriptors. The manifest is published
after event/image integrity checks. Requested actions, executed primitives,
public summaries, observations, errors, rejections, retries and outcomes remain
recorded. A malformed tool call is not a successful action; an ordinary rejected
submission is not automatically an infrastructure or recording failure.

`record_video=True`, or the launcher's `--record-video`, enables an owned-context
silent WebM; the default is off. The main-page video is finalized before manifest
publication, with one video across campaign navigation. `video.status` is
`not_requested`, `saved`, `missing` or `failed`. `saved` means nonempty finalized
bytes were found and hashed, not that playback was decoded or visually checked.
The recorder has no audio track. Event timestamps bracket its lifecycle but do
not establish a precisely synchronized video time zero.

`recording_complete` has scope `events_and_png_images`; video completeness is
reported separately. Missing video is missing video evidence, not proof that the
game failed. Discrete screenshots must not be presented as continuous recording.
An abrupt kill may leave partial events without a final manifest. Repeated
attempts use new directories, not repairs that erase earlier failures.

Keep outcome, evaluated status, CLI completion, recording, video and cleanup
separate. Local `infra_error` truncates and is excluded from model success/failure
grading, even with a complete archive. A bounded final state read catches late
local failures; already settled game results are preserved. A finish claim alone
cannot award reward. An orderly CLI exit or `evaluated=true` is not game success:
valid failures, voluntary stops and budget-limited attempts can be evaluated.
Launcher interruption, unmatched client configuration, incomplete recording or
cleanup, missing implementations and infrastructure failures are not valid wins.

The local client is limited to the reviewed game tools; its read-only shell
sandbox and disabled unrelated tools are not an OS sandbox for the MCP server.
Server code runs outside that client sandbox. Client model/effort evidence checks
the matching configured attempt, not provider-side attestation. Private reasoning
is neither requested nor fabricated. These archives are not yet a canonical
training-ready LiteSample/Parquet export or production env-server integration.

## Verification and remaining evidence

GUI regressions may use known inputs, visible DOM geometry and a virtual browser
clock as test oracles, while delivering real keyboard/pointer events. They verify
mechanics, not screenshot-only model ability or original timing equivalence.
Real-time checks and full video decoding establish different facts. Keep test
fixtures, frozen runtime versions, model settings and reference instances explicit;
do not combine selected successes from different runs into a single-run score.

For a fidelity claim, prioritize same-instance accepted/rejected boundaries,
normal-UI dynamic recordings, Refresh/reset behavior and lawful missing assets.
Source recovery can resolve rules without proving audiovisual or random-stream
parity. A successful end-to-end model attempt does not remove those limits, and
a failed attempt does not by itself identify an implementation bug. Do not freeze
games, widen hit windows or substitute answer replay merely to obtain a pass.

All local game traffic stays on the owned loopback service, which is not a public
authenticated server. The separate remote prototype has narrower grading and
access-policy limits, described in the README. Real access gates require stopping,
not bypassing them; no new original-site access or camera authorization follows
from local testing. Full task fidelity, per-task model coverage, complete recording
and an actual ordered campaign remain separate acceptance requirements.
