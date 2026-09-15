# Replacement, motion and unavailable-camera reference paths

This increment adds levels 18 and 22 and the explicitly limited unavailable-camera
path of level 39. All run locally through the existing registration, input,
observation, state and recording owners. They do not contact the original site.
Use [SUPPORT.md](/examples/not_a_robot/SUPPORT.md) for setup and
[VALIDATION.md](/examples/not_a_robot/VALIDATION.md) for actual test/model outcomes.

## Level 18: independent replacement images

The successful `attempt_401` captures an initial 3×3 board, seven stable round
states and the final pre-submit board. Only eight separate source photos exist;
they do not cover the replacement sequence. The implementation therefore uses
CSS crops of the **nine unchanged same-attempt JPEGs**, imported privately as
`level18_state00.jpg` through `level18_state08.jpg`. Original source paths and
hashes are pinned in [reference_manifest.json](/examples/not_a_robot/reference_manifest.json).
The first frame has a different parent scroll offset, so its crop origins differ.
The low-resolution reference quality is retained rather than replacing unseen
photos with invented artwork. Parent UI, old answers and completion logs are not
part of the displayed game.

Each cell has its own sequence of actual pictures and each picture has a captured
target/non-target label. A target click starts replacement of that cell only;
the other cells keep their current images. Verify requires no visible target,
no pending replacement and no selected distractor. The captured instance happens
to take 30 valid target clicks. Thirty background/repeated/wrong clicks cannot
pass, and the evaluator does not use a global click threshold.

Local choices not recovered from the original include:

- 650 ms replacement delay and a dimmed/selected transition state;
- ignoring duplicate clicks on an image already being replaced;
- independently reordering the per-cell sequences;
- toggling non-target selections and rejecting Verify while they remain selected;
- refreshing the same captured instance and cancelling all pending replacements.

These choices are explicit limitations, not original behavior proven by a video.
Required supplements are normal-play replacement video, incorrect/repeated-click
feedback, fresh instances and the missing photos at usable resolution.

## Level 22: nine separate moving entities

The successful `attempt_302` shows nine roaming ducks that individually return to
their 3×3 homes. Verify after the ninth return produces a matched official
completion record. The reference operator used DOM-assisted force-clicks with no
measured actual coordinates. That is implementation evidence, **not** a clean
screenshot-only tracking success.

The original private assets are a transparent 1023×341 three-frame duck sheet and
a 375×250 net. The reconstructed motion space is 600×590 pixels around a 460-pixel
card: ducks may cross the heading/footer and move into the right margin. They are
not confined to nine static tiles. Every entity has its own `roaming`, `returning`
and `caught` state; Verify requires all nine to be caught. A second click on a
returning/caught entity cannot advance another duck.

The following local parameters are fixed **before model testing**, not tuned to
obtain a pass:

| Parameter | Local value / policy |
| --- | --- |
| Visible input rectangle | 100×100 px per duck, ordinary browser hit testing |
| Initial positions/headings | Seed-derived, not recovered original randomness |
| Speed | 100–160 px/s per entity |
| Boundary movement | Reflected paths, evaluated from actual elapsed monotonic time |
| Sprite frame duration | 160 ms, three original frames |
| Return to assigned cell | 300 ms |
| Net | 100×66.7 px pointer graphic with inferred 27/20 px hotspot |

The game continues during model inference and screenshot acquisition. There is no
time freeze, click snapping, nearest-duck selection or acceptance based on old
screenshots. Capturing all nine through normal input may be difficult for a model
whose observation-to-action delay is large. A failure must remain a recorded
failure; it is not permission to slow the game or alter its hitboxes afterward.
The original speed, trajectory, overlap rules, transparent-pixel hitboxes, return
timing, sprite sequence and net collision shape remain unmeasured.

Required supplements are continuous normal-play movement/capture/return video,
missed clicks, incomplete Verify, refresh behavior and measured geometry/timing.

## Level 39: only the observed unavailable-device path

`attempt_501` displayed `0% HAPPY` and the camera-enable prompt while the capture
parent prohibited camera access. The record contains no camera permission request,
no face media and no camera-enable button. A normal Verify click nevertheless
produced a matched official completion message; afterward the displayed meter
was still zero. The reviewed clarification explicitly preserves that outcome
without claiming successful facial recognition.

The local page reproduces exactly that **limited branch**: heading, zero meter,
blank unavailable-camera pane/prompt and refresh/Verify controls. It requests no
device, stream, permission, face image or synthetic replacement face. Its success
reason is `reference_camera_unavailable_path_completed`, not an emotion score or
verified human identity. The reference metadata explicitly sets:

```text
rules_status = captured_unavailable_device_path
camera_flow_tested = false
facial_expression_detection_verified = false
human_identity_verified = false
```

Real-camera behavior, expression thresholds and rejection feedback remain
unimplemented. Any future camera experiment would need separate informed user
authorization and a clear retention policy; this task does not acquire it.

## Input calibration and limits

[test_input_timing.py](/examples/not_a_robot/tests/test_input_timing.py) observes
genuine browser keyboard and pointer events through test-only listeners. It checks
that key holds survive observations and a no-tool interval, release stops the held
state, and mouse press/move/release and the existing 20-segment drag remain actual
continuous input. Reset destroys the prior document and all test listeners.
The model receives none of this test-only DOM instrumentation.

This checks the runtime mechanism needed by later driving/drawing tasks. It does
not recover vehicle physics, drawing scorers or rhythm timing. Each child action
still incurs recording and screenshot costs, and `key_down` is a held key state,
not a promised stream of OS keyboard auto-repeat events. Further task-specific
input calibration remains necessary.
