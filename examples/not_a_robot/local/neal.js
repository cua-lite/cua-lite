/* Evidence-bounded reconstructions of ten synthetic game instances.
 * The screenshot-only policy does not receive this script or evaluator state.
 * Fixed answers identify captured instances, not a recovered original generator.
 * Timing, refresh, audio, and the tic-tac-toe opponent are documented local rules.
 */
"use strict";

window.renderNealTask = async ({task, seed, version}) => {
  const level = Number(task.id.slice(-2));
  document.body.classList.add("neal-mode");
  const root = document.createElement("section");
  root.className = "neal-game";
  root.dataset.level = String(level);
  root.dataset.status = "loading";
  root.setAttribute("aria-label", task.title);
  document.querySelector("main").append(root);

  const asset = (name) => `/reference_assets/${name}`;
  const imageNames = {
    2: ["level02_background.webp"],
    4: Array.from({length: 9}, (_, index) => `level04_image_${String(index + 1).padStart(2, "0")}.webp`),
    5: ["level05_intersection.webp"],
    8: ["level08_reference.jpg"],
    9: ["level09_background.webp"],
    10: ["level10_grass.webp", "level10_mole.png"],
  };
  await Promise.all((imageNames[level] || []).map(async (name) => {
    const image = new Image();
    image.src = asset(name);
    try {
      await image.decode();
    } catch {
      throw new Error(`Required private reference asset could not be decoded: ${name}. Run the asset importer.`);
    }
  }));

  const startTime = performance.now();
  const state = {task_id: task.id, label: task.title, version, seed,
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
      emit("retry", {reason: "refresh_same_reference_instance", revision});
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

  if (level === 1) {
    root.classList.add("neal-checkbox-card");
    const checkbox = button("neal-checkbox-target", "I'm not a robot");
    checkbox.setAttribute("role", "checkbox");
    checkbox.setAttribute("aria-checked", "false");
    const mark = document.createElement("span");
    mark.className = "neal-checkbox-mark";
    const caption = document.createElement("span");
    caption.textContent = "I'm not a robot";
    checkbox.append(mark, caption);
    const logo = document.createElement("div");
    logo.className = "neal-recaptcha-mark";
    logo.innerHTML = '<svg viewBox="0 0 36 36" aria-hidden="true"><path fill="#6083c5" d="M31 14A14 14 0 0 0 7 6L4 3v13h13l-5-5a8 8 0 0 1 14 3h5Z"/><path fill="#5264a3" d="M5 22a14 14 0 0 0 24 8l3 3V20H19l5 5a8 8 0 0 1-14-3H5Z"/></svg><span>reCAPTCHA</span>';
    root.append(checkbox, logo);
    let loading = false;
    checkbox.addEventListener("click", () => {
      if (!active() || loading) return;
      loading = true;
      mark.classList.add("loading");
      emit("checkbox_loading");
      // A loading phase is observed; its 650 ms duration is an inferred parameter.
      setTimeout(() => {
        if (!active()) return;
        mark.classList.remove("loading");
        mark.classList.add("checked");
        checkbox.setAttribute("aria-checked", "true");
        state.progress = 1;
        finish("checkbox_checked");
      }, 650);
    });
  } else if (level === 2 || level === 4 || level === 7) {
    const subjects = {2: "Stop Sign", 4: "Vegetable", 7: "Stop Sign and Bike"};
    heading("Select all the squares with a", subjects[level]);
    const size = level === 2 ? 4 : level === 4 ? 3 : 10;
    const expected = new Set(level === 2 ? [2, 3, 6, 7] : level === 4 ? [1, 2, 5] :
      [87, 76, 65, 54, 43, 32, 21, 10, 95, 96, 97, 98]);
    const letters = ["UILIBEWESD", "NPVBVFGBUK", "YGVOOLKOYW", "KBIRHOYNDD", "CSUSGUASGP",
      "LHFAPCWLWO", "KITIYOCWPN", "PMLUBJTFEJ", "JNWIYDVSRA", "IBUGUBIKEK"].join("");
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
    };
    verifyFooter(() => {
      if (exactSelection(selected, expected)) finish("exact_reference_selection");
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
    const answer = level === 3 ? "YHRPCD" : "867V 309";
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!active()) return;
      emit("submit", {input_length: input.value.length});
      if (input.value === answer) {
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
      photo.src = asset("level08_reference.jpg");
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
    refresh = () => initial.forEach((rotation, index) => {
      rotations[index] = rotation;
      faces[index].style.transform = `rotate(${rotation * 90}deg)`;
    });
    verifyFooter(() => {
      if (rotations.every((rotation) => rotation % 4 === 0)) finish("intersection_reassembled");
      else reject("intersection_not_reassembled");
    });
  } else if (level === 6) {
    heading("Win at", "Tic Tac Toe");
    const grid = newGrid(3);
    grid.classList.add("neal-tic-grid");
    const board = Array(9).fill("");
    board[4] = "O";
    const lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
    const hasWon = (mark) => lines.some((line) => line.every((index) => board[index] === mark));
    const winningMove = (mark) => {
      for (let index = 0; index < 9; index++) {
        if (board[index]) continue;
        board[index] = mark;
        const wins = hasWon(mark);
        board[index] = "";
        if (wins) return index;
      }
      return -1;
    };
    const tiles = [];
    const faces = [];
    let opponentPending = false;
    let opponentTimer = null;
    const redraw = () => board.forEach((mark, index) => {
      faces[index].className = `neal-tile-face${mark ? ` neal-mark-${mark.toLowerCase()}` : ""}`;
      tiles[index].setAttribute("aria-pressed", String(mark === "X"));
    });
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
          if (!active() || pendingRevision !== revision) return;
          // Inferred candidate fitting both recorded games, not recovered original AI.
          let move = winningMove("O");
          if (move < 0) move = winningMove("X");
          if (move < 0) move = [0, 2, 6, 8, 1, 3, 5, 7].find((position) => !board[position]);
          if (move !== undefined && move >= 0) {
            board[move] = "O";
            emit("move", {index: move, mark: "O"});
          }
          opponentPending = false;
          redraw();
          if (hasWon("O") || board.every(Boolean)) {
            emit("board_finished", {outcome: hasWon("O") ? "opponent_win" : "draw"});
          }
        }, 350);
      });
      tiles.push(tile);
      faces.push(face);
      grid.append(tile);
    }
    redraw();
    refresh = () => {
      clearTimeout(opponentTimer);
      opponentPending = false;
      board.fill("");
      board[4] = "O";
      redraw();
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
      if (exactSelection(selected, expected)) finish("exact_recursive_reference_selection");
      else reject("recursive_selection_does_not_match_reference_instance");
    });
  } else if (level === 10) {
    heading("Select all the squares with a", "Mole");
    const grid = newGrid(4);
    grid.classList.add("neal-mole-grid");
    const selected = new Set();
    const visibleUntil = Array(16).fill(0);
    const tiles = [];
    let nextSpawn = performance.now() + 700;
    let timer = null;
    let verify;
    const redraw = (now) => tiles.forEach((tile, index) => {
      tile.classList.toggle("mole-visible", visibleUntil[index] > now);
      tile.setAttribute("aria-pressed", String(selected.has(index)));
    });
    updateDynamic = () => {
      if (!active()) return;
      const now = performance.now();
      visibleUntil.forEach((until, index) => {
        if (until > 0 && until <= now) {
          visibleUntil[index] = 0;
          emit("mole_hidden", {index});
        }
      });
      if (now >= nextSpawn && selected.size < 5) {
        const candidates = Array.from({length: 16}, (_, index) => index)
          .filter((index) => !selected.has(index) && !visibleUntil[index]);
        if (candidates.length) {
          const index = candidates[Math.floor(random() * candidates.length)];
          // Real-time intervals are explicit approximations: no continuous video exists.
          const duration = 1200 + Math.floor(random() * 800);
          visibleUntil[index] = now + duration;
          emit("mole_shown", {index, duration_ms: duration});
        }
        nextSpawn = now + 850 + Math.floor(random() * 500);
      }
      redraw(now);
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
        updateDynamic();
        if (selected.has(index)) {
          emit("ignored_click", {index, reason: "already_hit"});
        } else if (selected.size >= 5) {
          emit("ignored_click", {index, reason: "hit_requirement_reached"});
        } else if (visibleUntil[index] > performance.now()) {
          selected.add(index);
          visibleUntil[index] = 0;
          state.progress = selected.size;
          emit("mole_hit", {index, hit_count: selected.size});
          verify.disabled = selected.size < 5;
          if (selected.size === 5) {
            visibleUntil.fill(0);
            clearInterval(timer);
          }
          redraw(performance.now());
        } else reject("no_visible_mole_at_click");
      });
      tiles.push(tile);
      grid.append(tile);
    }
    verify = verifyFooter(() => {
      if (selected.size === 5) finish("five_moles_hit_and_verified");
      else reject("mole_hit_requirement_not_reached");
    });
    verify.disabled = true;
    timer = setInterval(updateDynamic, 40);
    refresh = () => {
      selected.clear();
      visibleUntil.fill(0);
      nextSpawn = performance.now() + 700;
      verify.disabled = true;
      redraw(performance.now());
      clearInterval(timer);
      timer = setInterval(updateDynamic, 40);
    };
    stopDynamic = () => clearInterval(timer);
  } else {
    throw new Error(`Unsupported reference level: ${level}`);
  }

  root.dataset.status = "in_progress";
  Object.defineProperty(window, "syntheticTask", {value: Object.freeze({
    snapshot: () => {
      updateDynamic();
      return {...state, elapsed_ms: Math.round((finishedAt ?? performance.now()) - startTime),
        events: events.map((event) => ({...event}))};
    },
  }), writable: false, configurable: false});
  emit("ready");
};
