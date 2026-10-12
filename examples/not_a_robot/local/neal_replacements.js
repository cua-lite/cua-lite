/* L18's successful captured instance: each clicked hydrant gets its own next
 * photograph. Timing and distractor selection are explicit local policies.
 * Original photos are CSS crops of unchanged private reference screenshots.
 */
"use strict";

{
  const assets = Array.from({length: 9}, (_, index) =>
    `level18_state${String(index).padStart(2, "0")}.jpg`);
  const capturedFrames = [
    [0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0],
    [0, 1, 2, 3, 4, 5], [0], [0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2],
  ];
  // Visual review identifies every intermediate image in these five queues as
  // a hydrant and the final image as a distractor. Single-image cells never move.
  const queues = capturedFrames.map((frames) => frames.map((frame, index) => ({
    frame, isTarget: frames.length > 1 && index < frames.length - 1,
  })));
  const replacementDelayMs = 650;
  const columns = [[18, 82], [104, 82], [190, 82]];
  const initialRows = [[399, 83], [489, 83], [578, 84]];
  const stableRows = [[384, 84], [474, 84], [563, 84]];

  window.nealTasks[18] = {
    assets,
    render({asset, heading, newGrid, newTile, verifyFooter,
      state, emit, active, finish, reject}) {
      heading("Please select all the squares with a", "Fire Hydrant");
      const grid = newGrid(3);
      grid.classList.add("neal-replacement-grid");
      const cells = queues.map(() => ({position: 0, selected: false, timer: null}));
      const tiles = [];
      const remainingTargets = () => cells.filter((cell, index) =>
        queues[index][cell.position].isTarget).length;
      const drawCell = (index) => {
        const cell = cells[index];
        const picture = queues[index][cell.position];
        const {tile, face} = tiles[index];
        const [x, width] = columns[index % 3];
        const [y, height] = (picture.frame === 0 ? initialRows : stableRows)[Math.floor(index / 3)];
        face.style.backgroundImage = `url("${asset(assets[picture.frame])}")`;
        face.style.backgroundSize = `${304 / width * 100}% ${1211 / height * 100}%`;
        face.style.backgroundPosition = `${x / (304 - width) * 100}% ${y / (1211 - height) * 100}%`;
        face.style.backgroundRepeat = "no-repeat";
        face.style.opacity = cell.timer === null ? "1" : "0.35";
        tile.setAttribute("aria-pressed", String(cell.selected));
        tile.setAttribute("aria-busy", String(cell.timer !== null));
      };
      for (let index = 0; index < 9; index++) {
        const nodes = newTile(index);
        nodes.tile.setAttribute("aria-label", `Row ${Math.floor(index / 3) + 1}, column ${index % 3 + 1}`);
        nodes.tile.addEventListener("click", () => {
          if (!active()) return;
          const cell = cells[index];
          if (cell.timer !== null) {
            emit("ignored_click", {index, reason: "replacement_pending"});
            return;
          }
          const picture = queues[index][cell.position];
          if (!picture.isTarget) {
            // Wrong-cell behavior was not captured. This local policy retains
            // the photo, allows deselection, and rejects the extra selection.
            cell.selected = !cell.selected;
            drawCell(index);
            emit("selection_changed", {index, selected: cell.selected});
            return;
          }
          cell.selected = true;
          emit("selection_changed", {index, selected: true});
          emit("replacement_started", {index, from_frame: picture.frame,
            delay_ms: replacementDelayMs, timing_source: "local_unmeasured_parameter"});
          cell.timer = setTimeout(() => {
            cell.timer = null;
            if (!active()) return;
            const fromFrame = queues[index][cell.position].frame;
            cell.position++;
            cell.selected = false;
            state.progress = cells.reduce((sum, current) => sum + current.position, 0);
            drawCell(index);
            emit("image_replaced", {index, from_frame: fromFrame,
              to_frame: queues[index][cell.position].frame,
              remaining_hydrants: remainingTargets()});
          }, replacementDelayMs);
          drawCell(index);
        });
        tiles.push(nodes);
        grid.append(nodes.tile);
        drawCell(index);
      }
      verifyFooter(() => {
        if (cells.some((cell) => cell.timer !== null)) reject("replacement_pending");
        else if (cells.some((cell, index) => cell.selected && !queues[index][cell.position].isTarget)) {
          reject("distractor_selected");
        } else if (remainingTargets() !== 0) reject("hydrants_remain_visible");
        else finish("no_hydrants_or_pending_replacements");
      });
      const stopDynamic = () => {
        for (const cell of cells) {
          if (cell.timer !== null) clearTimeout(cell.timer);
          cell.timer = null;
        }
      };
      return {stopDynamic, refresh() {
        const canceled = cells.filter((cell) => cell.timer !== null).length;
        stopDynamic();
        cells.forEach((cell, index) => {
          cell.position = 0;
          cell.selected = false;
          drawCell(index);
        });
        state.progress = 0;
        emit("replacement_reset", {canceled_replacements: canceled});
      }};
    },
  };
}
