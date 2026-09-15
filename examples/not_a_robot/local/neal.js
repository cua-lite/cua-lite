/* Evidence-bounded reconstructions of synthetic game instances.
 * The screenshot-only policy does not receive this script or evaluator state.
 * Fixed answers identify captured instances, not a recovered original generator.
 * Per-task provenance distinguishes source-derived rules and local differences.
 */
"use strict";

window.nealTasks = {};

const ticTacToeLines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const ticTacToeMove = (board, gameNum, random) => {
  const empty = board.flatMap((mark, position) => mark ? [] : [position]);
  if (!empty.length) return -1;
  // Source 1121 consumes this chance draw even on the first, tactical-only game.
  if (random() < 0.4 && gameNum > 0) return empty[Math.floor(random() * empty.length)];
  for (const mark of ["O", "X"]) {
    for (const line of ticTacToeLines) {
      const open = line.find((index) => !board[index]);
      if (open !== undefined && line.filter((index) => board[index] === mark).length === 2) return open;
    }
  }
  for (const index of [4, 0, 2, 6, 8]) if (!board[index]) return index;
  return empty[0];
};

window.renderNealTask = async ({task, seed, version, referenceInstance}) => {
  const level = Number(task.id.slice(-2));
  document.body.classList.add("neal-mode");
  const root = document.createElement("section");
  root.className = "neal-game";
  root.dataset.level = String(level);
  root.dataset.referenceInstance = referenceInstance;
  root.dataset.status = "loading";
  root.setAttribute("aria-label", task.title);
  document.querySelector("main").append(root);

  const asset = (name) => `/reference_assets/${name}`;
  const imageNames = {
    2: ["level02_background.webp"],
    4: Array.from({length: 9}, (_, index) => `level04_image_${String(index + 1).padStart(2, "0")}.webp`),
    5: ["level05_intersection.webp"],
    8: [referenceInstance === "incremental" ? "level08_incremental_reference.jpg" : "level08_reference.jpg"],
    9: ["level09_background.webp"],
    10: ["level10_grass.webp", "level10_mole.png"],
  };
  await Promise.all((window.nealTasks[level]?.assets || imageNames[level] || []).map(async (name) => {
    const image = new Image();
    image.src = asset(name);
    try {
      await image.decode();
    } catch {
      throw new Error(`Required private reference asset could not be decoded: ${name}. Run the asset importer.`);
    }
  }));

  const startTime = performance.now();
  const state = {task_id: task.id, label: task.title, version, seed, reference_instance: referenceInstance,
    status: "in_progress", mistakes: 0, progress: 0, reason: ""};
  const events = [];
  let finishedAt = null;
  let revision = 0;
  let refresh = () => {};
  let updateDynamic = () => {};
  let stopDynamic = () => {};
  let randomState = seed >>> 0;
  const random = () => {
    randomState = (randomState + 0x6D2B79F5) >>> 0;
    let value = randomState;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
  const emit = (kind, details = {}) => events.push({sequence: events.length + 1,
    elapsed_ms: Math.round(performance.now() - startTime), kind, ...details});
  const active = () => state.status === "in_progress";
  const finish = (reason) => {
    if (!active()) return;
    state.status = "success";
    state.reason = reason;
    finishedAt = performance.now();
    root.dataset.status = "success";
    stopDynamic();
    root.querySelectorAll("button, input").forEach((node) => { node.disabled = true; });
    emit("success", {reason});
  };
  const reject = (reason) => {
    if (!active()) return;
    state.mistakes++;
    emit("rejected", {reason});
    // The recordings show no reliable rejection text; keep the visible board.
  };
  const button = (className, label) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = className;
    node.setAttribute("aria-label", label);
    return node;
  };
  const newRefresh = () => {
    const node = button("neal-refresh", "Refresh challenge");
    node.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7a8 8 0 0 0-14-1L3 9m0-5v5h5M4 17a8 8 0 0 0 14 1l3-3m0 5v-5h-5"/></svg>';
    node.addEventListener("click", () => {
      if (!active()) return;
      revision++;
      state.progress = 0;
      emit("retry", {reason: "refresh_challenge", revision});
      refresh();
    });
    return node;
  };
  const heading = (prefix, subject) => {
    const header = document.createElement("header");
    header.className = "neal-heading";
    const small = document.createElement("div");
    small.textContent = prefix;
    const large = document.createElement("strong");
    large.textContent = subject;
    header.append(small, large);
    root.append(header);
  };
  const newGrid = (size) => {
    const grid = document.createElement("div");
    grid.className = "neal-grid";
    grid.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${size}, minmax(0, 1fr))`;
    root.append(grid);
    return grid;
  };
  const newTile = (index) => {
    const tile = button("neal-tile", `Tile ${index + 1}`);
    tile.setAttribute("aria-pressed", "false");
    const face = document.createElement("span");
    face.className = "neal-tile-face";
    tile.append(face);
    return {tile, face};
  };
  const verifyFooter = (onVerify) => {
    const footer = document.createElement("div");
    footer.className = "neal-footer";
    const verify = button("neal-submit", "Verify");
    verify.textContent = "Verify";
    verify.addEventListener("click", () => {
      if (!active()) return;
      updateDynamic();
      emit("submit", {selected_count: state.progress});
      onVerify();
    });
    footer.append(newRefresh(), verify);
    root.append(footer);
    return verify;
  };
  const exactSelection = (selected, expected) => selected.size === expected.size &&
    [...selected].every((index) => expected.has(index));
  const wireCheckbox = ({card, checkbox, mark, wrong = false, details = {}, reason}) => {
    const timers = new Set();
    const spinner = document.createElement("span");
    spinner.className = "neal-checkbox-spinner";
    mark.append(spinner);
    card.style.cursor = "pointer";
    card.addEventListener("click", () => {
      if (!active()) return;
      // Source 478 permits repeated card clicks without clearing earlier marks
      // or callbacks. A previous click can end loading for a later click.
      mark.classList.add("loading");
      emit("checkbox_loading", details);
      const feedbackTimer = setTimeout(() => {
        timers.delete(feedbackTimer);
        if (!active()) return;
        mark.classList.remove("loading");
        mark.classList.add(wrong ? "wrong" : "checked");
        checkbox.setAttribute("aria-checked", "true");
        if (!wrong) state.progress = 1;
        emit(wrong ? "checkbox_wrong" : "checkbox_checked", details);
      }, wrong ? 800 : 700);
      timers.add(feedbackTimer);
      if (!wrong) {
        const completionTimer = setTimeout(() => {
          timers.delete(completionTimer);
          finish(reason);
        }, 1600);
        timers.add(completionTimer);
      }
    });
    // Local terminal cleanup suppresses stale callbacks after the single result.
    return () => {
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
      mark.classList.remove("loading");
    };
  };

  if (window.nealTasks[level]) {
    const hooks = await window.nealTasks[level].render({root, task, level, asset, button, heading,
      newGrid, newTile, verifyFooter, exactSelection, wireCheckbox, state, emit, active, finish, reject, random});
    refresh = hooks.refresh;
    if (hooks.stopDynamic) stopDynamic = hooks.stopDynamic;
  } else if (level === 1) {
    root.classList.add("neal-checkbox-card");
    const checkbox = button("neal-checkbox-target", "I'm not a robot");
    checkbox.setAttribute("role", "checkbox");
    checkbox.setAttribute("aria-checked", "false");
    const markContainer = document.createElement("span");
    markContainer.className = "neal-checkbox-mark-container";
    const mark = document.createElement("span");
    mark.className = "neal-checkbox-mark";
    markContainer.append(mark);
    const caption = document.createElement("span");
    caption.className = "neal-checkbox-caption";
    caption.textContent = "I'm not a robot";
    checkbox.append(markContainer, caption);
    const logo = document.createElement("div");
    logo.className = "neal-recaptcha-mark";
    logo.innerHTML = '<svg viewBox="0 0 36 36" aria-hidden="true"><path fill="#6083c5" d="M31 14A14 14 0 0 0 7 6L4 3v13h13l-5-5a8 8 0 0 1 14 3h5Z"/><path fill="#5264a3" d="M5 22a14 14 0 0 0 24 8l3 3V20H19l5 5a8 8 0 0 1-14-3H5Z"/></svg><span>reCAPTCHA</span>';
    root.append(checkbox, logo);
    stopDynamic = wireCheckbox({card: root, checkbox, mark, reason: "checkbox_checked"});
  } else if (level === 2 || level === 4 || level === 7) {
    const subjects = {2: "Stop Sign", 4: "Vegetable", 7: "Stop Sign and Bike"};
    heading("Select all the squares with a", subjects[level]);
    const size = level === 2 ? 4 : level === 4 ? 3 : 10;
    const expected = new Set(level === 2 ? [2, 3, 6, 7] : level === 4 ? [1, 2, 5, 7] :
      referenceInstance === "incremental" ? [26, 36, 46, 56, 66, 76, 86, 96, 77, 75, 74] :
      [87, 76, 65, 54, 43, 32, 21, 10, 95, 96, 97, 98]);
    const letters = (referenceInstance === "incremental" ?
      ["PCJYGSFLYT", "FFIWSNMENV", "PLSIWDSUCL", "BSSOGOTHUB", "JOEWVEOWFE",
        "PMYJDFPOLH", "LDFFRGSESM", "GNMWEKIBOL", "FFFCRAGSTO", "BPMTONNGAY"] :
      ["UILIBEWESD", "NPVBVFGBUK", "YGVOOLKOYW", "KBIRHOYNDD", "CSUSGUASGP",
        "LHFAPCWLWO", "KITIYOCWPN", "PMLUBJTFEJ", "JNWIYDVSRA", "IBUGUBIKEK"]).join("");
    const selected = new Set();
    const tiles = [];
    const grid = newGrid(size);
    if (level === 7) grid.classList.add("neal-letters");
    for (let index = 0; index < size * size; index++) {
      const {tile, face} = newTile(index);
      if (level === 2) {
        face.style.backgroundImage = `url("${asset("level02_background.webp")}")`;
        face.style.backgroundSize = "400% 400%";
        face.style.backgroundPosition = `${index % 4 * 100 / 3}% ${Math.floor(index / 4) * 100 / 3}%`;
      } else if (level === 4) {
        face.style.backgroundImage = `url("${asset(imageNames[4][index])}")`;
      } else {
        face.textContent = letters[index];
        tile.setAttribute("aria-label", `Row ${Math.floor(index / 10) + 1}, column ${index % 10 + 1}, ${letters[index]}`);
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
    refresh = () => {
      selected.clear();
      tiles.forEach((tile) => tile.setAttribute("aria-pressed", "false"));
      if (level === 7) {
        const generated = Array(100).fill("");
        const directions = [[1, 0], [-1, 0], [0, 1], [0, -1],
          [1, 1], [1, -1], [-1, 1], [-1, -1]];
        expected.clear();
        // Source 1072 sorts the words alphabetically, then samples maximum-overlap
        // candidates in direction order and row-major order within each direction.
        for (const word of ["BIKE", "STOPSIGN"]) {
          let bestOverlap = 0;
          let candidates = [];
          for (const [dx, dy] of directions) {
            for (let y = 0; y < 10; y++) {
              for (let x = 0; x < 10; x++) {
                const endX = x + dx * (word.length - 1);
                const endY = y + dy * (word.length - 1);
                if (endX < 0 || endX >= 10 || endY < 0 || endY >= 10) continue;
                const cells = Array.from(word, (_, index) => (y + dy * index) * 10 + x + dx * index);
                let overlap = 0;
                let conflict = false;
                for (let index = 0; index < word.length; index++) {
                  const letter = generated[cells[index]];
                  if (letter === word[index]) overlap++;
                  else if (letter) { conflict = true; break; }
                }
                if (conflict || overlap < bestOverlap) continue;
                if (overlap > bestOverlap) { bestOverlap = overlap; candidates = []; }
                candidates.push(cells);
              }
            }
          }
          // BIKE occupies at most four rows, leaving an empty row for STOPSIGN.
          // Thus the generic source generator's retry/growth branches are unreachable here.
          const placement = candidates[Math.floor(random() * candidates.length)];
          placement.forEach((cell, index) => {
            generated[cell] = word[index];
            expected.add(cell);
          });
        }
        const alphabet = "abcdefghijklmnoprstuvwy";
        tiles.forEach((tile, index) => {
          if (!generated[index]) generated[index] = alphabet[Math.floor(random() * 23)];
          const letter = generated[index].toUpperCase();
          tile.firstElementChild.textContent = letter;
          tile.setAttribute("aria-label", `Row ${Math.floor(index / 10) + 1}, column ${index % 10 + 1}, ${letter}`);
        });
        emit("word_search_regenerated");
      }
    };
    verifyFooter(() => {
      if (level === 4) {
        const missing = [...expected].filter((index) => !selected.has(index)).length;
        const extra = [...selected].filter((index) => !expected.has(index) && index !== 8).length;
        if (missing + extra <= 1) finish("vegetable_selection_within_tolerance");
        else reject("vegetable_selection_exceeds_tolerance");
      } else if (exactSelection(selected, expected)) finish("exact_reference_selection");
      else reject("selection_does_not_match_reference_instance");
    });
  } else if (level === 3 || level === 8) {
    root.classList.add("neal-text-card");
    const form = document.createElement("form");
    const prompt = document.createElement("label");
    prompt.htmlFor = "neal-answer";
    prompt.textContent = level === 3 ? "Enter the text below" : "Enter the perpetrator's license plate";
    const picture = document.createElement("div");
    picture.className = "neal-text-picture";
    const inputRow = document.createElement("div");
    inputRow.className = "neal-input-row";
    const input = document.createElement("input");
    input.id = "neal-answer";
    input.placeholder = "Answer";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.setAttribute("autocapitalize", "off");
    const submit = button("neal-submit", "Submit");
    submit.type = "submit";
    submit.textContent = "Submit";
    inputRow.append(input, submit);
    form.append(prompt, picture, inputRow);
    root.append(form);
    const answer = level === 3 ? "YHRPCD" : referenceInstance === "incremental" ? "JHB007" : "867V309";
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!active()) return;
      emit("submit", {input_length: input.value.length});
      // The source plate rule removes only ASCII hyphens and spaces, preserving case.
      const submitted = level === 8 ? input.value.replaceAll("-", "").replaceAll(" ", "") : input.value;
      if (submitted === answer) {
        state.progress = 1;
        finish("reference_text_accepted");
      } else reject("text_does_not_match_reference_instance");
    });
    input.addEventListener("input", () => {
      if (active()) emit("text_edited", {input_length: input.value.length});
    });
    let animationFrame = null;
    let animationStart = performance.now();
    if (level === 3) {
      const canvas = document.createElement("canvas");
      canvas.width = 428;
      canvas.height = 160;
      canvas.setAttribute("aria-label", "Animated distorted text");
      picture.append(canvas);
      const context = canvas.getContext("2d");
      const source = document.createElement("canvas");
      source.width = 428;
      source.height = 160;
      const ink = source.getContext("2d");
      ink.fillStyle = "#386d88";
      ink.font = "bold 110px Georgia, serif";
      ink.textBaseline = "alphabetic";
      const textWidth = ink.measureText(answer).width;
      ink.save();
      ink.translate(12, 0);
      ink.scale(400 / textWidth, 1);
      ink.fillText(answer, 0, 120);
      ink.restore();
      const lines = Array.from({length: 13}, () => ({x: random() * 428, y: random() * 160,
        dx: random() * 450 - 225, dy: random() * 300 - 150, bend: random() * 150 - 75}));
      const draw = (now) => {
        const phase = (now - animationStart) / 700;
        context.fillStyle = "#faf9f2";
        context.fillRect(0, 0, 428, 160);
        context.strokeStyle = "#c57458";
        context.lineWidth = 2.6;
        for (const line of lines) {
          context.beginPath();
          context.moveTo(line.x - line.dx, line.y - line.dy);
          context.quadraticCurveTo(line.x + line.bend, line.y + line.bend,
            line.x + line.dx, line.y + line.dy);
          context.stroke();
        }
        // Warp the full text raster by scanline; this is not independent glyph bobbing.
        for (let y = 0; y < 160; y++) {
          const displacement = 19 * Math.sin(y / 21 + phase) + 6 * Math.sin(y / 11 - phase / 2);
          context.drawImage(source, 0, y, 428, 1, displacement, y, 428, 1);
        }
        if (active()) animationFrame = requestAnimationFrame(draw);
      };
      draw(performance.now());
      stopDynamic = () => cancelAnimationFrame(animationFrame);
      const audio = button("neal-audio", "Audio unavailable in this local reference");
      audio.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 4V5L7 9H3Z" fill="currentColor" stroke="none"/><path d="M15 8q5 4 0 8M18 5q8 7 0 14"/></svg>';
      audio.addEventListener("click", () => {
        if (active()) emit("unsupported", {reason: "audio_not_present_in_reference_capture"});
      });
      picture.append(audio);
    } else {
      picture.classList.add("neal-plate-picture");
      const crop = document.createElement("div");
      crop.className = "neal-plate-crop";
      const photo = document.createElement("img");
      photo.src = asset(imageNames[8][0]);
      photo.alt = "Rear of a car and its license plate";
      photo.draggable = false;
      crop.append(photo);
      picture.append(crop);
    }
    picture.append(newRefresh());
    refresh = () => {
      input.value = "";
      animationStart = performance.now();
    };
  } else if (level === 5) {
    heading("Reassemble the", "Intersection");
    const grid = newGrid(3);
    grid.classList.add("neal-rotation-grid");
    // Accepted relative clicks were [3,3,0,0,0,3,0,0,0] quarter-turns.
    const initial = [1, 1, 0, 0, 0, 1, 0, 0, 0];
    const rotations = [...initial];
    const faces = [];
    for (let index = 0; index < 9; index++) {
      const {tile, face} = newTile(index);
      tile.removeAttribute("aria-pressed");
      face.style.backgroundImage = `url("${asset("level05_intersection.webp")}")`;
      face.style.backgroundSize = "300% 300%";
      face.style.backgroundPosition = `${index % 3 * 50}% ${Math.floor(index / 3) * 50}%`;
      face.style.transform = `rotate(${rotations[index] * 90}deg)`;
      tile.addEventListener("click", () => {
        if (!active()) return;
        rotations[index]++;
        face.style.transform = `rotate(${rotations[index] * 90}deg)`;
        // Progress counts actual clicks, not hidden correctly oriented tiles.
        state.progress++;
        emit("tile_rotated", {index, direction: "clockwise", quarter_turns: 1});
      });
      faces.push(face);
      grid.append(tile);
    }
    refresh = () => {
      // Source 1111 samples each tile independently, in grid order.
      faces.forEach((face, index) => {
        rotations[index] = Math.floor(4 * random());
        face.style.transform = `rotate(${rotations[index] * 90}deg)`;
      });
      emit("rotations_regenerated");
    };
    verifyFooter(() => {
      if (rotations.every((rotation) => rotation % 4 === 0)) finish("intersection_reassembled");
      else reject("intersection_not_reassembled");
    });
  } else if (level === 6) {
    heading("Win at", "Tic Tac Toe");
    const grid = newGrid(3);
    grid.classList.add("neal-tic-grid");
    const board = Array(9).fill("");
    const hasWon = (mark) => ticTacToeLines.some((line) => line.every((index) => board[index] === mark));
    const tiles = [];
    const faces = [];
    let gameNum = 0;
    let opponentPending = true;
    let opponentTimer = null;
    const redraw = () => board.forEach((mark, index) => {
      faces[index].className = `neal-tile-face${mark ? ` neal-mark-${mark.toLowerCase()}` : ""}`;
      tiles[index].setAttribute("aria-pressed", String(mark === "X"));
    });
    const playOpponent = () => {
      if (!active()) return;
      const move = ticTacToeMove(board, gameNum, random);
      if (move >= 0) {
        board[move] = "O";
        emit("move", {index: move, mark: "O"});
      }
      opponentPending = false;
      redraw();
      if (hasWon("O") || board.every(Boolean)) {
        emit("board_finished", {outcome: hasWon("O") ? "opponent_win" : "draw"});
      }
    };
    for (let index = 0; index < 9; index++) {
      const {tile, face} = newTile(index);
      tile.addEventListener("click", () => {
        if (!active()) return;
        if (opponentPending || board[index] || hasWon("O") || hasWon("X")) {
          emit("ignored_click", {index, reason: opponentPending ? "opponent_turn" : board[index] ? "occupied_cell" : "game_over"});
          return;
        }
        board[index] = "X";
        state.progress++;
        emit("move", {index, mark: "X"});
        redraw();
        if (hasWon("X") || board.every(Boolean)) return;
        opponentPending = true;
        const pendingRevision = revision;
        opponentTimer = setTimeout(() => {
          if (pendingRevision === revision) playOpponent();
        }, 450);
      });
      tiles.push(tile);
      faces.push(face);
      grid.append(tile);
    }
    redraw();
    opponentTimer = setTimeout(playOpponent, 100);
    refresh = () => {
      clearTimeout(opponentTimer);
      opponentPending = false;
      board.fill("");
      gameNum++;
      // Keep safe cancellation instead of the source's stale-timer refresh race.
      redraw();
      emit("board_reset", {first_player: "X"});
    };
    stopDynamic = () => clearTimeout(opponentTimer);
    verifyFooter(() => {
      if (hasWon("X")) finish("tic_tac_toe_player_win");
      else reject(opponentPending ? "opponent_turn" : hasWon("O") ? "opponent_win" : board.every(Boolean) ? "draw_not_win" : "no_player_win");
    });
  } else if (level === 9) {
    heading("Select all the squares with a", "Stop Sign");
    const board = document.createElement("div");
    board.className = "neal-recursive-board";
    board.style.backgroundImage = `url("${asset("level09_background.webp")}")`;
    root.append(board);
    // Integer coordinates are depth-four cells in the photo, not the parent screenshot.
    const acceptedRows = {4: [7, 8, 9, 10], 5: [7, 8, 9, 10, 11],
      6: [6, 7, 8, 9, 10, 11], 7: [6, 7, 8, 9, 10, 11],
      8: [6, 7, 8, 9, 10, 11], 9: [7, 8, 9, 10]};
    const expected = new Set(Object.entries(acceptedRows).flatMap(([row, columns]) =>
      columns.map((column) => Number(row) * 16 + column)));
    const selected = new Set();
    const addLeaf = (x, y, width, depth) => {
      const leaf = button("neal-recursive-leaf", `Image region, depth ${depth}`);
      leaf.style.left = `${x * 100}%`;
      leaf.style.top = `${y * 100}%`;
      leaf.style.width = `${width * 100}%`;
      leaf.style.height = `${width * 100}%`;
      leaf.setAttribute("aria-pressed", "false");
      leaf.addEventListener("click", () => {
        if (!active()) return;
        if (depth < 4) {
          leaf.remove();
          const half = width / 2;
          for (const [dx, dy] of [[0, 0], [half, 0], [0, half], [half, half]]) {
            addLeaf(x + dx, y + dy, half, depth + 1);
          }
          emit("region_split", {x, y, width, depth});
        } else {
          const index = Math.round(y * 16) * 16 + Math.round(x * 16);
          if (selected.has(index)) selected.delete(index);
          else selected.add(index);
          leaf.setAttribute("aria-pressed", String(selected.has(index)));
          state.progress = selected.size;
          emit("selection_changed", {x, y, width, depth, selected: selected.has(index), selected_count: selected.size});
        }
      });
      board.append(leaf);
    };
    addLeaf(0, 0, 1, 0);
    refresh = () => {
      selected.clear();
      board.replaceChildren();
      addLeaf(0, 0, 1, 0);
    };
    verifyFooter(() => {
      // Source 1100 counts missing and extra terminal leaves together.
      const errors = [...expected].filter((index) => !selected.has(index)).length
        + [...selected].filter((index) => !expected.has(index)).length;
      if (errors <= 2) finish("recursive_source_selection");
      else reject("recursive_selection_does_not_match_reference_instance");
    });
  } else if (level === 10) {
    heading("Select all the squares with a", "Mole");
    const grid = newGrid(4);
    grid.classList.add("neal-mole-grid");
    const selected = new Set();
    const whacked = new Set();
    const visible = new Set();
    const tiles = [];
    const hideTimers = new Set();
    let latestHideTimer = null;
    let spawnTimer = null;
    let verify;
    const redraw = () => {
      tiles.forEach((tile, index) => {
        tile.classList.toggle("mole-visible", visible.has(index));
        tile.classList.toggle("mole-whacked", whacked.has(index));
        tile.setAttribute("aria-pressed", String(selected.has(index)));
      });
      state.progress = whacked.size;
      verify.disabled = whacked.size < 5;
    };
    const stopTimers = () => {
      clearTimeout(spawnTimer);
      hideTimers.forEach((timer) => clearTimeout(timer));
      hideTimers.clear();
    };
    const showRandomMole = () => {
      if (!active()) return;
      // Source 1127 uses 1..16 even though Grid 379 renders only cells 0..15.
      const candidates = Array.from({length: 16}, (_, index) => index + 1)
        .filter((index) => !whacked.has(index) && !visible.has(index));
      if (!candidates.length) return;
      const index = candidates[Math.floor(random() * candidates.length)];
      const duration = [1500, 1000, 750, 625, 600][whacked.size];
      visible.add(index);
      emit("mole_shown", {index, duration_ms: duration});
      const hideTimer = setTimeout(() => {
        hideTimers.delete(hideTimer);
        if (active() && !whacked.has(index) && visible.delete(index)) {
          emit("mole_hidden", {index});
          redraw();
        }
      }, duration);
      hideTimers.add(hideTimer);
      latestHideTimer = hideTimer;
      spawnTimer = setTimeout(showRandomMole, 500 + 2000 * random());
      redraw();
    };
    for (let index = 0; index < 16; index++) {
      const {tile, face} = newTile(index);
      face.style.backgroundImage = `url("${asset("level10_grass.webp")}")`;
      const mole = document.createElement("span");
      mole.className = "neal-mole";
      // The original 400 x 200 sprite contains normal and struck 200 x 200 frames.
      mole.style.backgroundImage = `url("${asset("level10_mole.png")}")`;
      mole.setAttribute("aria-hidden", "true");
      face.append(mole);
      tile.addEventListener("click", () => {
        if (!active()) return;
        if (selected.has(index)) selected.delete(index);
        else selected.add(index);
        emit("selection_changed", {index, selected: selected.has(index), selected_count: selected.size});
        if (whacked.delete(index)) emit("mole_unhit", {index, hit_count: whacked.size});
        if (visible.has(index) && selected.has(index)) {
          whacked.add(index);
          visible.delete(index);
          emit("mole_hit", {index, hit_count: whacked.size});
          if (whacked.size >= 5) {
            // The source cancels only its latest hide and next-spawn handles here.
            clearTimeout(latestHideTimer);
            hideTimers.delete(latestHideTimer);
            clearTimeout(spawnTimer);
          }
        }
        redraw();
      });
      tiles.push(tile);
      grid.append(tile);
    }
    verify = verifyFooter(() => {
      if (whacked.size >= 5) finish("five_moles_hit_and_verified");
      else reject("mole_hit_requirement_not_reached");
    });
    refresh = () => {
      // Refresh/shutdown cancel every owned hide, not the original stale timers.
      stopTimers();
      selected.clear();
      whacked.clear();
      visible.clear();
      showRandomMole();
    };
    refresh();
    stopDynamic = stopTimers;
  } else {
    throw new Error(`Unsupported reference level: ${level}`);
  }

  root.dataset.status = state.status;
  if (task.reference.fidelity === "local_variant") {
    const note = document.createElement("p");
    note.className = "neal-variant-note";
    note.textContent = "Local mechanic variant. Artwork, media or dynamics differ from the original game.";
    root.append(note);
  }
  Object.defineProperty(window, "syntheticTask", {value: Object.freeze({
    snapshot: () => {
      updateDynamic();
      return {...state, elapsed_ms: Math.round((finishedAt ?? performance.now()) - startTime),
        events: events.map((event) => ({...event}))};
    },
  }), writable: false, configurable: false});
  emit("ready");
};
