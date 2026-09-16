/* Captured image-selection instances, not a recovered image generator.
 * Private images retain their original bytes. Level 12 uses CSS crops from its
 * successful attempt because the separate image bundle omits the muffin tiles.
 */
"use strict";

{
  const grids = {
    11: {
      prefix: "Select all the squares with", subject: "Waldo", columns: 25, rows: 25,
      assets: ["level11_background.webp"], expected: [218, 243], fine: true,
    },
    12: {
      prefix: "Select all the squares with a", subject: "Chihuahua", columns: 4, rows: 4,
      assets: ["level12_reference.jpg"], expected: [2, 3, 7, 10, 11, 15],
      screenshot: {width: 1280, height: 900,
        columns: [[20, 93], [117, 93], [214, 93], [310, 94]],
        rows: [[233, 92], [329, 93], [426, 93], [523, 93]]},
    },
    13: {
      prefix: "Select all the squares without a", subject: "Traffic Light", columns: 4, rows: 4,
      assets: ["level13_background.webp"], expected: [0, 3, 4, 7, 8, 11, 12, 13, 14, 15],
    },
    29: {
      prefix: "Select all the items with a", subject: "Soul", columns: 3, rows: 3,
      assets: Array.from({length: 9}, (_, index) =>
        `level29_image_${String(index + 1).padStart(2, "0")}.webp`),
      expected: [0, 5, 7], separateImages: true,
    },
    31: {
      prefix: "Select all the squares with a", subject: "Traffic Light", columns: 4, rows: 4,
      assets: ["level31_background.webp"],
      expected: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    },
    46: {
      prefix: "Select all the squares of the", subject: "64th Floor of the Empire State Building",
      columns: 14, rows: 57, assets: ["level46_background.webp"], fine: true,
      expected: [409, 410, 411, 412, 413, 414, 415, 416],
      imageAspectRatio: "899 / 3706", width: "586px",
    },
  };

  for (const [level, definition] of Object.entries(grids)) {
    window.nealTasks[level] = {
      assets: definition.assets,
      render({root, asset, heading, newGrid, newTile, verifyFooter,
        exactSelection, state, emit, active, finish, reject}) {
        const {columns, rows} = definition;
        if (definition.width) root.style.width = definition.width;
        if (level === "11") {
          root.classList.add("neal-waldo-card");
          // The source chooses the cell border at mount, independently of its CSS width breakpoint.
          root.style.setProperty("--neal-waldo-border", window.innerWidth < 800 ? "0.2px" : "1px");
        }
        heading(definition.prefix, definition.subject);
        const grid = newGrid(columns);
        grid.classList.add("neal-reference-selection");
        grid.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
        grid.style.aspectRatio = definition.imageAspectRatio || `${columns} / ${rows}`;
        if (definition.fine) {
          grid.classList.add("neal-fine-grid");
          grid.style.gap = level === "11" ? "0" : "1px";
        }
        const expected = new Set(definition.expected);
        const selected = new Set();
        const tiles = [];
        for (let index = 0; index < columns * rows; index++) {
          const {tile, face} = newTile(index);
          const column = index % columns;
          const row = Math.floor(index / columns);
          tile.setAttribute("aria-label", `Row ${row + 1}, column ${column + 1}`);
          face.style.backgroundRepeat = "no-repeat";
          if (definition.screenshot) {
            const source = definition.screenshot;
            const [x, width] = source.columns[column];
            const [y, height] = source.rows[row];
            face.style.backgroundImage = `url("${asset(definition.assets[0])}")`;
            face.style.backgroundSize = `${source.width / width * 100}% ${source.height / height * 100}%`;
            face.style.backgroundPosition = `${x / (source.width - width) * 100}% ${y / (source.height - height) * 100}%`;
          } else if (definition.separateImages) {
            face.style.backgroundImage = `url("${asset(definition.assets[index])}")`;
          } else {
            face.style.backgroundImage = `url("${asset(definition.assets[0])}")`;
            face.style.backgroundSize = `${columns * 100}% ${rows * 100}%`;
            face.style.backgroundPosition = `${column * 100 / (columns - 1)}% ${row * 100 / (rows - 1)}%`;
          }
          tile.addEventListener("click", () => {
            if (!active()) return;
            if (selected.has(index)) selected.delete(index);
            else selected.add(index);
            tile.setAttribute("aria-pressed", String(selected.has(index)));
            state.progress = selected.size;
            emit("selection_changed", {index, selected: selected.has(index), selected_count: selected.size});
          });
          tiles.push(tile);
          grid.append(tile);
        }
        verifyFooter(() => {
          let accepted;
          let reason = "exact_reference_selection";
          if (level === "11" || level === "46") {
            // Both source rules require every target but permit a bounded number of extras.
            accepted = [...expected].every((index) => selected.has(index))
              && selected.size <= (level === "11" ? 3 : 10);
            reason = level === "11" ? "waldo_source_selection" : "floor_source_selection";
          } else if (level === "29") {
            const errors = [...expected].filter((index) => !selected.has(index)).length
              + [...selected].filter((index) => !expected.has(index)).length;
            accepted = !selected.has(3) && errors <= 1;
            reason = "soul_source_selection";
          } else {
            accepted = exactSelection(selected, expected);
          }
          if (accepted) finish(reason);
          else reject("selection_does_not_match_reference_instance");
        });
        return {refresh() {
          selected.clear();
          tiles.forEach((tile) => tile.setAttribute("aria-pressed", "false"));
        }};
      },
    };
  }
}
