/* Independently authored implementations of source-derived game rules.
 * Original local audio/film in 32/47/48 is not original-site media fidelity.
 * Only the evaluator consumes task state; visual controllers receive pixels.
 */
"use strict";

// These three games own generated sound, gesture unlocking, pause and teardown.
function puzzleAudio(emit) {
  let context = null;
  let source = null;
  let track = null;
  let gain = null;
  let closed = false;
  return {
    async unlock() {
      if (closed) return false;
      if (!context) {
        context = new AudioContext();
        gain = context.createGain();
        gain.gain.value = 0.18;
        gain.connect(context.destination);
      }
      await context.resume();
      emit("audio_unlocked", {state: context.state, source: "local_web_audio"});
      return context.state === "running";
    },
    tone(frequency, duration = 0.28) {
      if (!context || closed || context.state !== "running") return;
      const oscillator = context.createOscillator();
      const envelope = context.createGain();
      oscillator.type = "triangle";
      oscillator.frequency.value = frequency;
      envelope.gain.setValueAtTime(0, context.currentTime);
      envelope.gain.linearRampToValueAtTime(0.5, context.currentTime + 0.02);
      envelope.gain.exponentialRampToValueAtTime(0.001, context.currentTime + duration);
      oscillator.connect(envelope).connect(gain);
      oscillator.start();
      oscillator.stop(context.currentTime + duration);
      oscillator.onended = () => { oscillator.disconnect(); envelope.disconnect(); };
      emit("audio_tone", {frequency_hz: frequency, duration_ms: duration * 1000});
    },
    compose(duration, notes) {
      // A mono PCM track composed here, not downloaded or copied from a recording.
      const rate = 11025;
      track = context.createBuffer(1, Math.ceil(duration * rate), rate);
      const pcm = track.getChannelData(0);
      for (const note of notes) {
        const start = Math.floor(note.time * rate);
        const length = Math.floor((note.duration || 0.24) * rate);
        for (let sample = 0; sample < length && start + sample < pcm.length; sample++) {
          const t = sample / rate;
          const envelope = Math.min(1, t / 0.015) * Math.exp(-t * 10);
          pcm[start + sample] += 0.35 * envelope * Math.sin(2 * Math.PI * note.frequency * t);
        }
      }
      emit("audio_composed", {duration_seconds: duration, notes: notes.length, sample_rate: rate,
        source: "original_local_composition"});
    },
    play(offset = 0) {
      if (!context || closed || !track) return;
      if (source) { source.onended = null; source.stop(); source.disconnect(); }
      source = context.createBufferSource();
      source.buffer = track;
      source.connect(gain);
      source.start(0, offset);
      source.onended = () => emit("audio_buffer_ended", {source: "original_local_composition"});
      emit("audio_playing", {offset_seconds: offset});
    },
    pause() {
      if (source) { source.onended = null; source.stop(); source.disconnect(); source = null; }
    },
    close() {
      closed = true;
      this.pause();
      if (context && context.state !== "closed") void context.close();
      emit("audio_closed");
    },
  };
}

window.nealTasks[20] = {
  // Only the captured initial image is available. Do not fabricate nine variants.
  assets: ["level20_image_01.webp"],
  render({root, asset, heading, verifyFooter, state, emit, active, finish, reject}) {
    heading("Describe what you see in the", "Inkblot");
    const image = document.createElement("img");
    image.className = "neal-inkblot";
    image.alt = "Symmetrical inkblot";
    const input = document.createElement("input");
    input.className = "neal-puzzle-answer";
    input.placeholder = "Answer";
    input.setAttribute("aria-label", "Answer");
    root.append(image, input);
    const draw = () => { image.src = asset(this.assets[0]); };
    const submit = () => {
      if (!active()) return;
      emit("text_submitted", {utf16_length: input.value.length});
      if (input.value.length > 5) { state.progress = 1; finish("description_length_accepted"); }
      else { reject("description_needs_six_utf16_units"); input.setAttribute("aria-invalid", "true"); }
    };
    input.addEventListener("input", () => input.removeAttribute("aria-invalid"));
    input.addEventListener("keydown", (event) => { if (event.key === "Enter") submit(); });
    verifyFooter(submit);
    draw();
    return {refresh() {
      input.value = "";
      input.removeAttribute("aria-invalid");
      draw();
      emit("inkblot_refreshed", {image_id: 1, fidelity: "original_nine_image_refresh_cycle_incomplete"});
    }};
  },
};

window.nealTasks[27] = {
  assets: [],
  render({root, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject}) {
    heading("Connect matching colors and", "Fill every square");
    const pairs = [[0, 27], [5, 15], [26, 28], [13, 3], [11, 35], [8, 14]];
    const colors = ["#ef4444", "#15b9cd", "#ed9028", "#e2c532", "#8d51c3", "#f473b3"];
    const endpoints = new Map(pairs.flatMap((pair, color) => pair.map((cell) => [cell, color])));
    const grid = newGrid(6);
    grid.classList.add("neal-network-grid");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 600 600");
    svg.classList.add("neal-network-lines");
    grid.append(svg);
    let paths = new Map();
    let connected = new Set();
    let dragging = null;
    const faces = [];
    const draw = () => {
      svg.replaceChildren();
      for (const [color, path] of paths) {
        const line = document.createElementNS(svg.namespaceURI, "polyline");
        line.setAttribute("points", path.map((cell) => `${cell % 6 * 100 + 50},${Math.floor(cell / 6) * 100 + 50}`).join(" "));
        line.setAttribute("stroke", colors[color]);
        line.setAttribute("fill", "none");
        line.setAttribute("stroke-width", "27");
        line.setAttribute("stroke-linejoin", "round");
        line.setAttribute("stroke-linecap", "round");
        svg.append(line);
      }
      faces.forEach((face, cell) => {
        const color = endpoints.get(cell);
        face.textContent = color === undefined ? "" : "●";
        face.style.color = color === undefined ? "transparent" : colors[color];
      });
      state.progress = connected.size;
      grid.dataset.connected = String(connected.size);
    };
    const end = () => {
      if (dragging === null) return;
      const path = paths.get(dragging);
      const pair = pairs[dragging];
      if (path.length > 1 && pair.includes(path[0]) && pair.includes(path.at(-1)) && path[0] !== path.at(-1)) {
        connected.add(dragging);
        emit("network_connected", {color: dragging, cells: [...path]});
      } else {
        paths.delete(dragging);
        emit("network_incomplete_path_cleared", {color: dragging});
      }
      dragging = null;
      draw();
    };
    const enter = (cell) => {
      if (!active() || dragging === null) return;
      const path = paths.get(dragging);
      const previous = path.at(-1);
      if (path.includes(cell) || Math.abs(previous % 6 - cell % 6) +
        Math.abs(Math.floor(previous / 6) - Math.floor(cell / 6)) !== 1) return;
      if ([...paths].some(([color, route]) => color !== dragging && route.includes(cell))) return;
      const endpoint = endpoints.get(cell);
      if (endpoint !== undefined && endpoint !== dragging) { end(); return; }
      path.push(cell);
      emit("network_path_extended", {color: dragging, cell});
      draw();
      if (endpoint !== undefined) end();
    };
    for (let cell = 0; cell < 36; cell++) {
      const {tile, face} = newTile(cell);
      tile.removeAttribute("aria-pressed");
      tile.dataset.cell = String(cell);
      tile.addEventListener("pointerdown", (event) => {
        if (!active() || event.button !== 0) return;
        event.preventDefault();
        end();
        let color = endpoints.get(cell);
        if (color === undefined) color = [...paths].find(([, route]) => route.includes(cell))?.[0];
        if (color === undefined) return;
        connected.delete(color);
        paths.set(color, [cell]);
        dragging = color;
        emit("network_path_started", {color, cell});
        draw();
      });
      tile.addEventListener("pointerenter", () => enter(cell));
      grid.append(tile);
      faces.push(face);
    }
    const touchMove = (event) => {
      if (event.pointerType !== "touch" || dragging === null) return;
      event.preventDefault();
      const tile = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-cell]");
      if (tile && grid.contains(tile)) enter(Number(tile.dataset.cell));
    };
    document.addEventListener("pointerup", end);
    grid.addEventListener("pointermove", touchMove);
    verifyFooter(() => {
      const occupied = new Set([...endpoints.keys(), ...[...paths.values()].flat()]);
      if (connected.size === 6 && occupied.size === 36) finish("all_network_pairs_cover_board");
      else reject("network_connections_or_coverage_incomplete");
    });
    draw();
    return {refresh() { paths = new Map(); connected = new Set(); dragging = null; draw(); },
      stopDynamic() { document.removeEventListener("pointerup", end); grid.removeEventListener("pointermove", touchMove); }};
  },
};

window.nealTasks[32] = {
  assets: [],
  render({root, button, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject, random}) {
    heading("Remember and repeat the", "Flashing sequence");
    const info = document.createElement("p");
    info.className = "neal-puzzle-info";
    const start = button("neal-submit", "Start with sound");
    start.textContent = "Start with sound";
    root.append(info, start);
    const grid = newGrid(5);
    grid.classList.add("neal-memory-grid");
    const pads = [];
    const audio = puzzleAudio(emit);
    let round = 0;
    let phase = "idle";
    let sequence = [];
    let entered = 0;
    let timer = null;
    const paint = () => {
      grid.dataset.phase = phase;
      grid.dataset.round = String(round + 1);
      pads.forEach((pad) => { pad.disabled = phase !== "input"; });
      info.textContent = phase === "input" ? `Your turn: ${entered}/${sequence.length}` :
        phase === "success" ? "All three sequences correct" : phase === "failed" ? "Wrong order. Starting again…" :
          phase === "idle" ? "Start to enable synthesized audio" : `Round ${round + 1}: watch ${sequence.length} pads`;
    };
    const frequency = (index) => 130.8128 * 2 ** (([0, 2, 4, 7, 9][index % 5] + Math.floor(index / 5) * 12) / 12);
    const flash = (position) => {
      if (!active()) return;
      if (position === sequence.length) { phase = "input"; paint(); return; }
      phase = "playing";
      const index = sequence[position];
      pads[index].classList.add("lit");
      audio.tone(frequency(index));
      emit("memory_pad_shown", {index, step: position, round: round + 1});
      paint();
      timer = setTimeout(() => {
        pads[index].classList.remove("lit");
        timer = setTimeout(() => flash(position + 1), 300);
      }, 600);
    };
    const next = () => {
      clearTimeout(timer);
      entered = 0;
      sequence = Array.from({length: 3 + round}, () => Math.floor(random() * 25));
      phase = "waiting";
      pads.forEach((pad) => pad.classList.remove("lit"));
      paint();
      timer = setTimeout(() => flash(0), 1500);
    };
    for (let index = 0; index < 25; index++) {
      const {tile, face} = newTile(index);
      face.style.background = `hsl(${index * 29 % 360} 65% 45%)`;
      tile.removeAttribute("aria-pressed");
      tile.addEventListener("click", () => {
        if (!active() || phase !== "input") return;
        audio.tone(frequency(index));
        emit("memory_input", {index, step: entered, round: round + 1});
        if (index !== sequence[entered]) {
          reject("memory_wrong_order");
          round = 0;
          state.progress = 0;
          phase = "failed";
          paint();
          timer = setTimeout(next, 1000);
          return;
        }
        entered++;
        if (entered === sequence.length) {
          state.progress = round + 1;
          if (round === 2) { phase = "success"; verify.disabled = false; paint(); }
          else { round++; next(); }
        } else paint();
      });
      pads.push(tile);
      grid.append(tile);
    }
    const verify = verifyFooter(() => {
      if (phase === "success") finish("memory_three_rounds_verified");
      else reject("memory_incomplete");
    });
    verify.disabled = true;
    root.querySelector(".neal-refresh").hidden = true;
    root.querySelector(".neal-refresh").style.display = "none";
    start.addEventListener("click", async () => {
      if (phase !== "idle" || !active()) return;
      start.disabled = true;
      try {
        if (!await audio.unlock()) throw new Error("Audio is not running");
        start.hidden = true;
        next();
      } catch (error) { info.textContent = `Audio could not start: ${error.message}`; start.disabled = false; }
    });
    paint();
    return {refresh: next, stopDynamic() { clearTimeout(timer); audio.close(); }};
  },
};

window.nealTasks[36] = {
  assets: [],
  render({root, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject, random}) {
    heading("Reach 1,000 points in", "Match Three");
    const info = document.createElement("p");
    info.className = "neal-puzzle-info";
    root.append(info);
    const grid = newGrid(8);
    grid.classList.add("neal-candy-grid");
    let board = [];
    let score = 0;
    let moves = 30;
    let selected = null;
    let busy = false;
    let gameOver = false;
    let timer = null;
    const tiles = [];
    const scan = () => {
      const matches = new Set();
      for (const vertical of [false, true]) {
        for (let outer = 0; outer < 8; outer++) {
          let start = 0;
          while (start < 8) {
            let end = start + 1;
            const index = (offset) => vertical ? offset * 8 + outer : outer * 8 + offset;
            while (end < 8 && board[index(end)] === board[index(start)]) end++;
            if (end - start >= 3) for (let i = start; i < end; i++) matches.add(index(i));
            start = end;
          }
        }
      }
      return matches;
    };
    const draw = () => {
      tiles.forEach(({tile, face}, index) => {
        face.className = `neal-tile-face candy-${board[index]}`;
        tile.setAttribute("aria-label", `Candy ${board[index]}, row ${Math.floor(index / 8) + 1}, column ${index % 8 + 1}`);
        tile.setAttribute("aria-pressed", String(index === selected));
        tile.disabled = busy || gameOver;
      });
      grid.dataset.phase = busy ? "animating" : gameOver ? "game-over" : "ready";
      info.textContent = score >= 1000 ? `Sweet victory! ${score} points` : moves === 0 ?
        `No moves left. ${score} / 1000 points. Refresh to retry.` : `${score} / 1000 points · ${moves} moves left`;
      info.dataset.score = String(score);
      info.dataset.moves = String(moves);
      state.progress = score;
    };
    const settle = (matches, depth = 0) => {
      if (!active()) return;
      if (!matches.size) {
        busy = false;
        gameOver = score >= 1000 || moves <= 0;
        if (gameOver) emit("candy_game_over", {score, moves, won: score >= 1000});
        draw();
        return;
      }
      const points = matches.size * 10 + Math.max(0, matches.size - 3) * 20;
      score += points;
      emit("candies_matched", {cells: [...matches], count: matches.size, points, score, cascade: depth});
      draw();
      matches.forEach((index) => tiles[index].tile.classList.add("matched"));
      timer = setTimeout(() => {
        for (let column = 0; column < 8; column++) {
          const remaining = Array.from({length: 8}, (_, row) => row * 8 + column)
            .filter((index) => !matches.has(index)).map((index) => board[index]);
          const missing = 8 - remaining.length;
          const fresh = Array(missing);
          for (let row = missing - 1; row >= 0; row--) fresh[row] = 1 + Math.floor(random() * 6);
          const rebuilt = [...fresh, ...remaining];
          for (let row = 0; row < 8; row++) board[row * 8 + column] = rebuilt[row];
        }
        tiles.forEach(({tile}) => tile.classList.remove("matched"));
        draw();
        timer = setTimeout(() => settle(scan(), depth + 1), 400);
      }, 200);
    };
    const swap = (first, second) => {
      if (!active() || busy || gameOver || Math.abs(first % 8 - second % 8) +
        Math.abs(Math.floor(first / 8) - Math.floor(second / 8)) !== 1) return;
      busy = true;
      selected = null;
      [board[first], board[second]] = [board[second], board[first]];
      emit("candies_swapped", {first, second});
      draw();
      timer = setTimeout(() => {
        const matches = scan();
        if (!matches.size) {
          [board[first], board[second]] = [board[second], board[first]];
          reject("candy_swap_has_no_match");
          draw();
          timer = setTimeout(() => { busy = false; draw(); }, 400);
        } else { moves--; settle(matches); }
      }, 400);
    };
    for (let index = 0; index < 64; index++) {
      const entry = newTile(index);
      let dragStart = null;
      let dragged = false;
      entry.tile.addEventListener("pointerdown", (event) => {
        if (!active() || busy || gameOver) return;
        dragStart = [event.clientX, event.clientY];
        dragged = false;
        entry.tile.setPointerCapture(event.pointerId);
      });
      entry.tile.addEventListener("pointerup", (event) => {
        if (!dragStart) return;
        const dx = event.clientX - dragStart[0];
        const dy = event.clientY - dragStart[1];
        dragStart = null;
        if (Math.max(Math.abs(dx), Math.abs(dy)) <= entry.tile.clientWidth / 4) return;
        dragged = true;
        const horizontal = Math.abs(dx) > Math.abs(dy);
        const target = index + (horizontal ? Math.sign(dx) : Math.sign(dy) * 8);
        if (target >= 0 && target < 64) swap(index, target);
      });
      entry.tile.addEventListener("click", () => {
        if (dragged) { dragged = false; return; }
        if (!active() || busy || gameOver) return;
        if (selected === null || selected === index) { selected = selected === index ? null : index; draw(); }
        else { const first = selected; selected = index; swap(first, index); draw(); }
      });
      entry.tile.addEventListener("keydown", (event) => {
        const step = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8}[event.key];
        if (step !== undefined) { event.preventDefault(); const target = index + step; if (target >= 0 && target < 64) swap(index, target); }
      });
      tiles.push(entry);
      grid.append(entry.tile);
    }
    const refresh = () => {
      clearTimeout(timer);
      score = 0; moves = 30; selected = null; busy = false; gameOver = false;
      board = Array.from({length: 64}, () => 1 + Math.floor(random() * 6));
      // Source removes initial runs by redrawing the whole matching set in scan order.
      let matches = scan();
      while (matches.size) {
        for (const index of matches) board[index] = 1 + Math.floor(random() * 6);
        matches = scan();
      }
      tiles.forEach(({tile}) => tile.classList.remove("matched"));
      draw();
    };
    verifyFooter(() => {
      if (score >= 1000) finish("candy_score_target_reached");
      else reject("candy_score_below_target");
    });
    refresh();
    return {refresh, stopDynamic() { clearTimeout(timer); }};
  },
};

window.nealTasks[37] = {
  assets: Array.from({length: 9}, (_, i) => `level37_image_${String(i + 1).padStart(2, "0")}.webp`),
  render({asset, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject, random}) {
    heading("Select the game-labeled", "AI-generated faces");
    const grid = newGrid(3);
    const tiles = [];
    let order = [];
    const selected = new Set();
    for (let index = 0; index < 9; index++) {
      const entry = newTile(index);
      entry.tile.addEventListener("click", () => {
        if (!active()) return;
        if (selected.has(index)) selected.delete(index); else selected.add(index);
        entry.tile.setAttribute("aria-pressed", String(selected.has(index)));
        state.progress = selected.size;
        emit("face_selection_changed", {index, selected: selected.has(index)});
      });
      tiles.push(entry);
      grid.append(entry.tile);
    }
    const refresh = () => {
      order = Array.from({length: 9}, (_, i) => i + 1).sort(() => random() - 0.5);
      selected.clear();
      tiles.forEach(({tile, face}, i) => {
        face.style.backgroundImage = `url("${asset(this.assets[order[i] - 1])}")`;
        tile.setAttribute("aria-pressed", "false");
      });
    };
    verifyFooter(() => {
      const labels = new Set([1, 6, 7, 9]);
      const chosen = new Set([...selected].map((index) => order[index]));
      const errors = [...labels].filter((id) => !chosen.has(id)).length + [...chosen].filter((id) => !labels.has(id)).length;
      if (errors <= 1) finish("face_labels_within_one_error"); else reject("face_label_error_count_exceeds_one");
    });
    refresh();
    return {refresh};
  },
};

window.nealTasks[40] = {
  assets: [],
  render({root, heading, verifyFooter, state, emit, active, finish, reject, random}) {
    heading("Type each symbol as it passes to", "Lock the five reels");
    const cabinet = document.createElement("div");
    cabinet.className = "neal-slot-cabinet";
    const input = document.createElement("input");
    input.className = "neal-puzzle-answer";
    input.setAttribute("aria-label", "Answer");
    input.autocomplete = "off";
    input.spellcheck = false;
    root.append(cabinet, input);
    const symbols = ["grape", "cherry", "watermelon", "diamond", "dollar", "banana"];
    const glyphs = {grape: "🍇", cherry: "🍒", watermelon: "🍉", diamond: "◆", dollar: "$", banana: "🍌"};
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const periods = [4500, 3750, 3000, 2250, 1500];
    let sequences = [];
    let answer = "";
    let lastInputValue = "";
    const reels = [];
    for (let column = 0; column < 5; column++) {
      const window = document.createElement("div");
      window.className = "neal-slot-window";
      const reel = document.createElement("div");
      reel.className = "neal-slot-reel";
      reel.style.animationDuration = `${periods[column]}ms`;
      window.append(reel);
      cabinet.append(window);
      reels.push(reel);
    }
    const refresh = () => {
      answer = ""; input.value = ""; lastInputValue = "";
      sequences = reels.map((reel) => {
        const special = Math.floor(random() * 5);
        const sequence = Array.from({length: 5}, (_, i) => i === special ?
          alphabet[Math.floor(random() * alphabet.length)] : symbols[Math.floor(random() * symbols.length)]);
        reel.replaceChildren();
        for (const value of [...sequence, ...sequence]) {
          const span = document.createElement("span");
          span.textContent = glyphs[value] || value;
          span.setAttribute("aria-label", value);
          reel.append(span);
        }
        reel.style.animationPlayState = "running";
        return sequence;
      });
    };
    input.addEventListener("input", () => {
      if (!active()) return;
      // The source wrapper calls onChange only when the input value changes.
      if (input.value === lastInputValue) return;
      lastInputValue = input.value;
      const length = input.value.length;
      if (length > 5) return;
      if (length < answer.length) {
        for (let i = answer.length - 1; i >= length; i--) {
          if (reels[i]) reels[i].style.animationPlayState = "running";
        }
        answer = answer.slice(0, length);
        emit("slot_suffix_unlocked", {length});
      } else if (length > 0) {
        const column = length - 1;
        reels[column].style.animationPlayState = "paused";
        const components = getComputedStyle(reels[column]).translate.split(/\s+/);
        const percentage = parseFloat(components[1] || "0");
        const position = Math.round(Math.abs(percentage) / 100 * 10) % 5;
        answer += sequences[column][position][0];
        emit("slot_reel_locked", {column, position});
      }
      state.progress = Math.min(answer.length, 5);
    });
    const submit = () => {
      if (!active()) return;
      if (answer.length === 5 && input.value.toLowerCase() === answer.toLowerCase()) finish("typed_symbols_match_frozen_reels");
      else reject("typed_symbols_do_not_match_frozen_reels");
    };
    input.addEventListener("keydown", (event) => { if (event.key === "Enter") submit(); });
    verifyFooter(submit);
    refresh();
    return {refresh, stopDynamic() { reels.forEach((reel) => { reel.style.animationPlayState = "paused"; }); }};
  },
};

window.nealTasks[47] = {
  assets: ["level47_cover.webp"],
  async render({root, asset, button, heading, verifyFooter, state, emit, active, finish, reject}) {
    const response = await fetch(asset("level47_chart.json"));
    if (!response.ok) throw new Error("Required private source chart is missing: level47_chart.json. Run the reference importer.");
    const directions = ["left", "down", "up", "right"];
    const chart = (await response.json()).map((entry) => ({time: entry.time, lane: directions.indexOf(entry.key)}));
    heading("Follow the arrows", "Rhythm with original chart");
    const note = document.createElement("p");
    note.className = "neal-puzzle-info";
    note.textContent = "Original-site 336-note chart and hit timing. Reach 85% over a 102-second locally synthesized media substitute, not the original song or video.";
    const start = button("neal-submit", "Start music");
    start.textContent = "Start music";
    const info = document.createElement("p");
    info.className = "neal-puzzle-info";
    const board = document.createElement("div");
    board.className = "neal-rhythm-board";
    board.style.backgroundImage = `linear-gradient(#17213b88, #17213baa), url("${asset("level47_cover.webp")}")`;
    const keys = ["ArrowLeft", "ArrowDown", "ArrowUp", "ArrowRight"];
    const glyphs = ["←", "↓", "↑", "→"];
    const targets = document.createElement("div");
    targets.className = "neal-rhythm-targets";
    const buttons = [];
    keys.forEach((key, lane) => {
      const node = button("neal-rhythm-target", key);
      node.textContent = glyphs[lane];
      targets.append(node);
      buttons.push(node);
    });
    board.append(targets);
    root.append(note, start, info, board);
    const audio = puzzleAudio((kind, details) => emit(kind, {...details,
      chart: "source_chart", media: "local_synthesized_media"}));
    const duration = 102;
    let alignAge = 1.78;
    let notes = [];
    let running = false;
    let completed = false;
    let startTime = 0;
    let hits = 0;
    let missed = 0;
    let frame = null;
    const pressed = new Set();
    let startedAudio = false;
    let round = 0;
    let closed = false;
    const timers = new Set();
    const visualTimers = new Set();
    const removeNote = (node) => {
      const timer = setTimeout(() => {
        visualTimers.delete(timer);
        node.remove();
      }, 300);
      visualTimers.add(timer);
    };
    const clearFeedback = () => {
      visualTimers.forEach((timer) => clearTimeout(timer));
      visualTimers.clear();
      notes.forEach((current) => current.node?.remove());
      pressed.clear();
      buttons.forEach((node) => node.classList.remove("pressed"));
    };
    const accuracy = () => hits + missed ? Math.round(hits * 100 / (hits + missed)) : 0;
    const drawInfo = () => {
      info.textContent = `${accuracy()}% · ${hits} hits · ${missed} misses${completed ? " · Complete, verify below" : ""}`;
      board.dataset.phase = running ? "playing" : completed ? "completed" : "idle";
      state.progress = hits;
    };
    const stopRound = () => {
      running = false;
      round++;
      cancelAnimationFrame(frame);
      frame = null;
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
      audio.pause();
      // Ending stops judgment, not the source-derived visual feedback already in flight.
    };
    const end = (early = false) => {
      stopRound();
      completed = !early && accuracy() >= 85;
      verify.disabled = !completed;
      emit("rhythm_ended", {early, hits, missed, accuracy: accuracy(), duration_seconds: (performance.now() - startTime) / 1000,
        chart: "source_chart", media: "local_synthesized_media",
        end_signal: early ? "accuracy_early_exit" : "local_synthesized_media_end", original_media_end_observed: false});
      if (!completed) { reject(early ? "rhythm_accuracy_below_eighty_percent" : "rhythm_accuracy_below_eighty_five_percent"); start.textContent = "Try again"; start.hidden = false; }
      else audio.tone(880, 0.5);
      drawInfo();
    };
    const update = (now) => {
      if (!running || !active()) return;
      const t = (now - startTime) / 1000;
      if (hits + missed > 20 && hits / (hits + missed) < 0.8) { end(true); return; }
      for (const current of notes) {
        if (!current.spawned && t >= current.time - alignAge) {
          current.spawned = true;
          current.spawnTime = now;
          current.node = document.createElement("span");
          current.node.className = "neal-rhythm-note";
          current.node.textContent = glyphs[current.lane];
          current.node.dataset.lane = String(current.lane);
          current.node.dataset.note = String(current.index);
          current.node.style.left = `calc(${current.lane * 25}% + 5px)`;
          current.node.addEventListener("animationend", () => {
            // A queued event from a refreshed/closed round must not own a new timer.
            if (current.node.isConnected) removeNote(current.node);
          }, {once: true});
          board.append(current.node);
          emit("rhythm_note_spawned", {lane: current.lane, note: current.index,
            chart_time_seconds: current.time, spawn_elapsed_seconds: t, align_age_seconds: alignAge});
          const currentRound = round;
          const timer = setTimeout(() => {
            timers.delete(timer);
            if (!running || !active() || currentRound !== round || current.judged) return;
            current.judged = true;
            missed++;
            current.node.classList.add("missed");
            emit("rhythm_missed", {note: current.index, lane: current.lane, reason: "late"});
            drawInfo();
          }, (alignAge + 0.25) * 1000);
          current.timer = timer;
          timers.add(timer);
        }
      }
      drawInfo();
      // This retained endpoint belongs to the synthesized local media variant,
      // not an observed ended event from the unavailable original video.
      if (t >= duration) { end(); return; }
      frame = requestAnimationFrame(update);
    };
    const hit = (lane) => {
      if (!active() || !running) return;
      let current = null;
      for (const candidate of notes) {
        if (candidate.lane === lane && candidate.spawned && !candidate.judged &&
            (!current || candidate.spawnTime < current.spawnTime)) current = candidate;
      }
      if (!current) return;
      const error = (performance.now() - current.spawnTime) / 1000 - alignAge;
      current.judged = true;
      clearTimeout(current.timer);
      timers.delete(current.timer);
      if (Math.abs(error) < 0.25) {
        hits++;
        current.node.classList.add("played");
        removeNote(current.node);
      } else {
        missed++;
        current.node.classList.add("missed");
      }
      emit("rhythm_judged", {note: current.index, lane, timing_error_seconds: error, hit: Math.abs(error) < 0.25});
      drawInfo();
    };
    const down = (event) => {
      const lane = keys.indexOf(event.key);
      if (lane < 0) return;
      event.preventDefault();
      if (pressed.has(event.key)) return;
      pressed.add(event.key);
      buttons[lane].classList.add("pressed");
      hit(lane);
    };
    const up = (event) => {
      const lane = keys.indexOf(event.key);
      if (lane < 0) return;
      event.preventDefault();
      pressed.delete(event.key);
      buttons[lane].classList.remove("pressed");
    };
    document.addEventListener("keydown", down);
    document.addEventListener("keyup", up);
    buttons.forEach((node, lane) => {
      node.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!active() || !running) return;
        node.setPointerCapture(event.pointerId);
        pressed.add(keys[lane]);
        node.classList.add("pressed");
        hit(lane);
      });
      const release = (event) => {
        event.preventDefault();
        event.stopPropagation();
        pressed.delete(keys[lane]);
        node.classList.remove("pressed");
      };
      node.addEventListener("pointerup", release);
      node.addEventListener("pointercancel", release);
    });
    const refresh = () => {
      stopRound();
      clearFeedback();
      completed = false; hits = 0; missed = 0;
      notes = [];
      start.hidden = false; start.disabled = false; start.textContent = "Start music";
      verify.disabled = true;
      drawInfo();
    };
    start.addEventListener("click", async () => {
      if (!active() || running) return;
      const requestedRound = round;
      start.disabled = true;
      try {
        if (!await audio.unlock()) throw new Error("Audio is not running");
        if (closed || requestedRound !== round || !active()) return;
        if (!startedAudio) {
          audio.compose(duration, chart.map((item) => ({time: item.time, frequency: [261.626, 329.628, 391.995, 523.251][item.lane]})));
          startedAudio = true;
        }
        refresh();
        const bounds = board.getBoundingClientRect();
        const target = buttons[0].getBoundingClientRect();
        const measuredAge = 2 * (target.top + target.height / 2 - bounds.top + 22.5) / (bounds.height + 45);
        alignAge = measuredAge > 0.5 && measuredAge < 2 ? measuredAge : 1.78;
        notes = chart.map((entry, index) => ({...entry, index, spawned: false, judged: false}));
        start.hidden = true;
        running = true;
        startTime = performance.now();
        audio.play();
        emit("rhythm_started", {duration_seconds: duration, note_count: notes.length, window_seconds: 0.25,
          align_age_seconds: alignAge, chart: "source_chart", media: "local_synthesized_media",
          clock: "performance_time_local_media_variant"});
        // The first nominal note predates alignAge: spawn it now, never backdate it.
        update(startTime);
      } catch (error) { info.textContent = `Audio could not start: ${error.message}`; start.disabled = false; }
    });
    const verify = verifyFooter(() => {
      if (completed) finish("source_chart_local_synthesized_media_verified"); else reject("rhythm_not_completed");
    });
    refresh();
    return {refresh, stopDynamic() {
      closed = true; stopRound(); clearFeedback(); audio.close();
      document.removeEventListener("keydown", down); document.removeEventListener("keyup", up);
    }};
  },
};

window.nealTasks[48] = {
  assets: [],
  render({root, button, heading, verifyFooter, state, emit, active, finish, reject}) {
    heading("One last moment", "An original local finale");
    const notice = document.createElement("p");
    notice.className = "neal-puzzle-info";
    notice.textContent = "70-second original animation and synthesized music. This is not Neal's original film or a certificate.";
    const canvas = document.createElement("canvas");
    canvas.width = 440; canvas.height = 330;
    canvas.className = "neal-finale-canvas";
    canvas.setAttribute("aria-label", "Original animated journey from a circuit to a flowering garden");
    const toggle = button("neal-submit", "Play finale");
    toggle.textContent = "Play finale";
    const time = document.createElement("p");
    time.className = "neal-puzzle-info";
    root.append(notice, canvas, toggle, time);
    const ink = canvas.getContext("2d");
    const audio = puzzleAudio(emit);
    let readyAudio = false;
    let playing = false;
    let elapsed = 0;
    let startedAt = 0;
    let frame = null;
    let ended = false;
    let confetti = false;
    const duration = 70;
    const captions = ["A pattern begins with a small idea.", "Curiosity creates new connections.",
      "Practice turns mistakes into new paths.", "A garden grows one choice at a time.",
      "Thank you for exploring this local world."];
    const draw = (seconds) => {
      const scene = Math.min(4, Math.floor(seconds / 14));
      ink.fillStyle = ["#11233d", "#173c50", "#284c40", "#35544a", "#173547"][scene];
      ink.fillRect(0, 0, 440, 330);
      for (let i = 0; i < 22; i++) {
        const x = 30 + i % 11 * 38;
        const y = 70 + Math.floor(i / 11) * 105 + 12 * Math.sin(seconds * 0.8 + i);
        ink.strokeStyle = `hsl(${130 + i * 7} 55% 65%)`;
        ink.lineWidth = 2;
        ink.beginPath();
        ink.moveTo(x, 265);
        ink.quadraticCurveTo(x + Math.sin(seconds + i) * 18, 180, x, y);
        ink.stroke();
        ink.fillStyle = `hsl(${i * 31 + seconds * 3} 70% 72%)`;
        ink.beginPath();
        ink.arc(x, y, 4 + scene * 2 + Math.sin(seconds + i) * 2, 0, Math.PI * 2);
        ink.fill();
      }
      ink.fillStyle = "#fff";
      ink.font = "15px Arial";
      ink.textAlign = "center";
      ink.fillText(captions[scene], 220, 305);
      if (seconds > 65) for (let i = 0; i < 55; i++) {
        ink.fillStyle = `hsl(${i * 57 % 360} 80% 65%)`;
        ink.fillRect((i * 83 + Math.sin(seconds + i) * 12) % 440, ((seconds - 65) * (45 + i % 30) + i * 7) % 280, 5, 8);
      }
      time.textContent = `${seconds.toFixed(1)} / 70.0 seconds${ended ? " · Playback complete" : playing ? " · Playing" : " · Paused"}`;
      canvas.dataset.phase = ended ? "ended" : playing ? "playing" : "paused";
      canvas.dataset.elapsed = seconds.toFixed(3);
      state.progress = Math.floor(seconds);
    };
    const animate = (now) => {
      if (!playing || !active()) return;
      const seconds = Math.min(duration, elapsed + (now - startedAt) / 1000);
      if (seconds > 65 && !confetti) {
        confetti = true;
        audio.tone(1046.5, 0.6);
        emit("finale_confetti", {time_seconds: seconds, source: "original_canvas_effect"});
      }
      if (seconds >= duration) {
        elapsed = duration; playing = false; ended = true;
        toggle.disabled = true; verify.disabled = false;
        audio.pause();
        emit("finale_animation_ended", {duration_seconds: duration, fidelity: "original_local_animation_music"});
      }
      draw(seconds);
      if (playing) frame = requestAnimationFrame(animate);
    };
    toggle.addEventListener("click", async () => {
      if (!active() || ended) return;
      if (playing) {
        elapsed += (performance.now() - startedAt) / 1000;
        playing = false; cancelAnimationFrame(frame); audio.pause();
        toggle.textContent = "Resume finale"; toggle.setAttribute("aria-label", "Resume finale");
        emit("finale_paused", {time_seconds: elapsed}); draw(elapsed); return;
      }
      toggle.disabled = true;
      try {
        if (!await audio.unlock()) throw new Error("Audio is not running");
        if (!readyAudio) {
          audio.compose(duration, Array.from({length: 140}, (_, i) => ({time: i * 0.5,
            frequency: [261.626, 293.665, 349.228, 391.995, 440][(i * 3 + Math.floor(i / 8)) % 5], duration: 0.45})));
          readyAudio = true;
        }
        audio.play(elapsed);
        playing = true; startedAt = performance.now();
        toggle.textContent = "Pause finale"; toggle.setAttribute("aria-label", "Pause finale");
        emit("finale_playing", {offset_seconds: elapsed, duration_seconds: duration,
          fidelity: "original_local_animation_music"});
        frame = requestAnimationFrame(animate);
      } catch (error) { time.textContent = `Audio could not start: ${error.message}`; }
      toggle.disabled = false;
    });
    const verify = verifyFooter(() => {
      if (ended) {
        const terminal = document.createElement("h2");
        terminal.className = "neal-local-terminal";
        terminal.textContent = "Local finale complete";
        root.append(terminal);
        finish("original_local_finale_played_and_verified");
      } else reject("finale_playback_not_ended");
    });
    const refresh = () => {
      cancelAnimationFrame(frame); audio.pause(); playing = false; elapsed = 0; ended = false; confetti = false;
      verify.disabled = true; toggle.disabled = false; toggle.textContent = "Play finale";
      toggle.setAttribute("aria-label", "Play finale"); draw(0);
      emit("finale_rewound");
    };
    refresh();
    return {refresh, stopDynamic() { playing = false; cancelAnimationFrame(frame); audio.close(); }};
  },
};
