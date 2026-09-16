/* Locally authored synthetic tasks, including documented mechanic adaptations.
 * Only the evaluator can read the status bridge through the CUA interface;
 * the CUA agent receives screenshots and mouse/keyboard tools, not JS access. */
"use strict";

(async () => {
  const catalog = await fetch("/tasks.json").then((response) => {
    if (!response.ok) throw new Error("Task catalog could not be loaded.");
    return response.json();
  });
  const query = new URLSearchParams(location.search);
  const seedText = query.get("seed") || "0";
  if (!/^\d+$/.test(seedText) || Number(seedText) > 4294967295) {
    throw new Error("Seed must be an integer from 0 through 4294967295.");
  }
  const seed = Number(seedText);
  document.getElementById("version").textContent = `v${catalog.version}`;
  const taskId = query.get("task");
  if (!taskId) {
    document.getElementById("gallery").hidden = false;
    document.getElementById("seed").value = seed;
    document.getElementById("seed-form").addEventListener("submit", (event) => {
      event.preventDefault();
      location.search = new URLSearchParams({seed: document.getElementById("seed").value});
    });
    const galleryTasks = [...catalog.tasks].sort((left, right) =>
      (left.id.startsWith("neal_") ? left.reference.level : 100) -
      (right.id.startsWith("neal_") ? right.reference.level : 100));
    galleryTasks.forEach((task, index) => {
      const link = document.createElement("a");
      link.className = "card";
      link.href = `/?${new URLSearchParams({task: task.id, seed})}`;
      const number = document.createElement("small");
      const kind = task.reference?.fidelity === "captured_instance" ? "REFERENCE" : "LOCAL VARIANT";
      number.textContent = task.reference ? `NEAL L${task.reference.level} / ${kind}` : `${index + 1} / ORIGINAL`;
      const title = document.createElement("h2");
      title.textContent = task.title;
      const description = document.createElement("p");
      description.textContent = task.instruction;
      link.append(number, title, description);
      document.getElementById("task-cards").append(link);
    });
    return;
  }
  const task = catalog.tasks.find((item) => item.id === taskId);
  if (!task) throw new Error(`Unknown task: ${taskId}`);
  const referenceInstance = query.get("instance") ?? "default";
  if (referenceInstance !== "default" && !Object.hasOwn(task.reference_variants || {}, referenceInstance)) {
    throw new Error(`Unknown reference instance ${referenceInstance} for ${taskId}`);
  }
  if (task.id.startsWith("neal_")) {
    await window.renderNealTask({task, seed, version: catalog.version, referenceInstance});
    return;
  }
  let randomState = seed >>> 0;
  const random = () => {
    randomState = (randomState + 0x6D2B79F5) >>> 0;
    let value = randomState;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
  const shuffled = (items) => {
    const result = [...items];
    for (let index = result.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1));
      [result[index], result[other]] = [result[other], result[index]];
    }
    return result;
  };
  const exercise = document.getElementById("exercise");
  exercise.hidden = false;
  exercise.dataset.task = task.id;
  exercise.dataset.status = "in_progress";
  document.getElementById("task-title").textContent = task.title;
  document.getElementById("task-number").textContent = `TASK 0${catalog.tasks.indexOf(task) + 1} / ${task.skill.toUpperCase()}`;
  document.getElementById("seed-label").textContent = `SEED ${seed} · REAL TIME`;
  const instruction = document.getElementById("instruction");
  instruction.textContent = task.instruction;
  if (task.reference) {
    const note = document.getElementById("reference-note");
    note.hidden = false;
    note.textContent = `Inspired by Neal's level ${task.reference.level}: ${task.reference.title}. Local artwork and rules; not an exact replica.`;
  }
  const board = document.getElementById("board");
  if (query.get("runner") !== "1") {
    document.getElementById("preview-controls").hidden = false;
    document.getElementById("restart").addEventListener("click", () => location.reload());
  }
  const startTime = performance.now();
  const state = {task_id: task.id, label: task.title, version: catalog.version, seed,
    status: "in_progress", mistakes: 0, progress: 0, reason: ""};
  const events = [];
  let finishedAt = null;
  const emit = (kind, details = {}) => events.push({sequence: events.length + 1,
    elapsed_ms: Math.round(performance.now() - startTime), kind, ...details});
  const feedback = (text) => { document.getElementById("feedback").textContent = text; };
  const mistake = (reason) => {
    if (state.status !== "in_progress") return;
    state.mistakes++;
    document.getElementById("mistakes").textContent = `Mistakes: ${state.mistakes}`;
    feedback(reason);
    emit("mistake", {reason});
  };
  const finish = (status, reason) => {
    if (state.status !== "in_progress") return;
    state.status = status;
    state.reason = reason;
    finishedAt = performance.now();
    exercise.dataset.status = status;
    feedback(status === "success" ? "Task complete. Nicely done." : "Time is up. Task failed.");
    board.querySelectorAll("button, input").forEach((element) => { element.disabled = true; });
    emit(status, {reason});
  };
  let refreshDeadline = () => {};
  Object.defineProperty(window, "syntheticTask", {value: Object.freeze({
    snapshot: () => {
      refreshDeadline();
      return {...state, elapsed_ms: Math.round((finishedAt ?? performance.now()) - startTime),
        events: events.map((event) => ({...event}))};
    }
  }), writable: false, configurable: false});
  emit("ready");

  if (task.id === "checkbox") {
    const panel = document.createElement("div");
    panel.className = "checkbox-panel";
    const card = document.createElement("div");
    card.className = "checkbox-card";
    const box = document.createElement("button");
    box.className = "checkbox-box";
    box.setAttribute("role", "checkbox");
    box.setAttribute("aria-checked", "false");
    box.setAttribute("aria-label", "I'm not a robot");
    const caption = document.createElement("div");
    caption.className = "checkbox-caption";
    caption.textContent = "I'm not a robot";
    const disclaimer = document.createElement("small");
    disclaimer.textContent = "Synthetic game only · No security verification";
    caption.append(disclaimer);
    box.addEventListener("click", () => {
      if (state.status !== "in_progress") return;
      box.setAttribute("aria-checked", "true");
      box.textContent = "✓";
      state.progress = 1;
      finish("success", "checkbox_checked");
    });
    card.append(box, caption);
    panel.append(card);
    board.append(panel);
  } else if (task.id === "stop_signs") {
    const types = shuffled(["stop", "stop", "stop", "stop", "yield", "yield", "parking", "tree", "tree"]);
    const targets = new Set(types.flatMap((type, index) => type === "stop" ? [index] : []));
    const selected = new Set();
    const grid = document.createElement("div");
    grid.className = "sign-grid";
    const count = document.createElement("p");
    count.textContent = "0 tiles selected";
    types.forEach((type, index) => {
      const tile = document.createElement("button");
      tile.className = "sign-tile";
      tile.setAttribute("aria-label", `Tile ${index + 1}`);
      tile.setAttribute("aria-pressed", "false");
      // All artwork is authored here; no photos or assets from the remote game.
      const offset = Math.floor(random() * 34) - 17;
      let object;
      if (type === "stop") {
        object = '<path d="M110 80V137" stroke="#777b7d" stroke-width="8"/>' +
          '<polygon points="92,12 128,12 148,32 148,68 128,88 92,88 72,68 72,32" fill="#c63332" stroke="white" stroke-width="4"/>' +
          '<text x="110" y="59" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold" fill="white">STOP</text>';
      } else if (type === "yield") {
        object = '<path d="M110 78V137" stroke="#777b7d" stroke-width="8"/>' +
          '<path d="M70 20H150L110 90Z" fill="white" stroke="#c63332" stroke-width="9"/>';
      } else if (type === "parking") {
        object = '<path d="M110 80V137" stroke="#777b7d" stroke-width="8"/>' +
          '<rect x="77" y="15" width="66" height="73" rx="5" fill="#3169c7" stroke="white" stroke-width="3"/>' +
          '<text x="110" y="68" text-anchor="middle" font-family="sans-serif" font-size="52" font-weight="bold" fill="white">P</text>';
      } else {
        object = '<path d="M110 74V137" stroke="#826244" stroke-width="12"/>' +
          '<circle cx="110" cy="45" r="34" fill="#4f9660"/><circle cx="87" cy="69" r="26" fill="#438552"/><circle cx="130" cy="66" r="27" fill="#5c9e67"/>';
      }
      tile.innerHTML = '<svg viewBox="0 0 220 140" aria-hidden="true">' +
        '<rect width="220" height="140" fill="#d9e9f0"/><path d="M0 93Q65 62 125 93T220 85V140H0Z" fill="#a9bd8d"/>' +
        '<path d="M0 123H220V140H0Z" fill="#7d8588"/>' +
        `<g transform="translate(${offset},0)">${object}</g></svg>`;
      tile.addEventListener("click", () => {
        if (state.status !== "in_progress") return;
        if (selected.has(index)) selected.delete(index);
        else selected.add(index);
        tile.setAttribute("aria-pressed", String(selected.has(index)));
        state.progress = selected.size;
        count.textContent = `${selected.size} tiles selected`;
        emit("selection_changed", {selected_count: selected.size});
      });
      grid.append(tile);
    });
    const verify = document.createElement("button");
    verify.className = "primary";
    verify.textContent = "Verify";
    verify.addEventListener("click", () => {
      if (state.status !== "in_progress") return;
      if (selected.size === targets.size && [...selected].every((index) => targets.has(index))) {
        finish("success", "exact_stop_sign_selection");
      } else {
        mistake("Selection does not match. Check all tiles and try again.");
      }
    });
    const row = document.createElement("div");
    row.className = "verify-row";
    row.append(count, verify);
    board.append(grid, row);
  } else if (task.id === "click") {
    const options = shuffled(["blue circle", "blue square", "blue triangle", "orange circle", "orange square", "orange triangle"]);
    const target = options[Math.floor(random() * options.length)];
    instruction.textContent = `Click the ${target}. Choose the matching color AND shape.`;
    const grid = document.createElement("div");
    grid.className = "shape-grid";
    options.forEach((option) => {
      const tile = document.createElement("button");
      tile.className = "shape-tile";
      const [color, shape] = option.split(" ");
      const icon = document.createElement("span");
      icon.className = `shape ${color} ${shape}`;
      icon.setAttribute("aria-hidden", "true");
      const label = document.createElement("span");
      label.textContent = option;
      tile.append(icon, label);
      tile.addEventListener("click", () => {
        if (state.status !== "in_progress") return;
        if (option === target) { state.progress = 1; finish("success", "matched"); }
        else mistake("That is not the requested tile. Try again.");
      });
      grid.append(tile);
    });
    board.append(grid);
  } else if (task.id === "input" || task.id === "wiggles") {
    const animated = task.id === "wiggles";
    const alphabet = animated ? "ABCDEFGHJKLMNPQRSTUVWXYZ" : "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    const code = Array.from({length: animated ? 6 : 5}, () => alphabet[Math.floor(random() * alphabet.length)]).join("");
    const form = document.createElement("form");
    form.className = "code-panel";
    const prompt = document.createElement("div");
    prompt.className = animated ? "code wiggle-code" : "code";
    if (animated) {
      for (const letter of code) {
        const glyph = document.createElement("span");
        glyph.textContent = letter;
        glyph.style.animationDuration = `${1.4 + random()}s`;
        glyph.style.animationDelay = `${-random() * 2}s`;
        prompt.append(glyph);
      }
    } else prompt.textContent = code;
    const label = document.createElement("label");
    label.htmlFor = "code-input";
    label.textContent = "Enter the code exactly as shown (uppercase).";
    const input = document.createElement("input");
    input.id = "code-input";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.maxLength = 32;
    const submit = document.createElement("button");
    submit.className = "primary";
    submit.textContent = "Submit";
    form.append(prompt, label, input, submit);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (state.status !== "in_progress") return;
      if (input.value === code) { state.progress = 1; finish("success", "exact_code"); }
      else mistake("The code does not match. Edit your entry and try again.");
    });
    board.append(form);
  } else if (task.id === "drag") {
    const arena = document.createElement("div");
    arena.className = "drag-arena";
    const slot = document.createElement("div");
    slot.className = "drop-slot";
    slot.textContent = "DROP HERE";
    const block = document.createElement("div");
    block.className = "block";
    block.setAttribute("aria-label", "Blue draggable block");
    arena.append(slot, block);
    board.append(arena);
    const initialX = 12 + Math.floor(random() * 30);
    const initialY = 30 + Math.floor(random() * 80);
    const slotX = Math.max(150, arena.clientWidth - 160 - Math.floor(random() * 35));
    const slotY = 20 + Math.floor(random() * 50);
    slot.style.left = `${slotX}px`;
    slot.style.top = `${slotY}px`;
    const resetBlock = () => { block.style.left = `${initialX}px`; block.style.top = `${initialY}px`; };
    resetBlock();
    let drag = null;
    block.addEventListener("pointerdown", (event) => {
      if (state.status !== "in_progress" || event.button !== 0 || drag) return;
      drag = {x: event.clientX, y: event.clientY, pointer: event.pointerId};
      block.setPointerCapture(event.pointerId);
      emit("drag_started");
    });
    block.addEventListener("pointermove", (event) => {
      if (!drag || drag.pointer !== event.pointerId) return;
      block.style.left = `${Math.max(0, Math.min(arena.clientWidth - 64, initialX + event.clientX - drag.x))}px`;
      block.style.top = `${Math.max(0, Math.min(arena.clientHeight - 64, initialY + event.clientY - drag.y))}px`;
    });
    block.addEventListener("pointerup", (event) => {
      if (!drag || drag.pointer !== event.pointerId) return;
      const moved = Math.hypot(event.clientX - drag.x, event.clientY - drag.y) >= 8;
      const piece = block.getBoundingClientRect();
      const target = slot.getBoundingClientRect();
      drag = null;
      block.releasePointerCapture(event.pointerId);
      if (moved && piece.left >= target.left && piece.right <= target.right && piece.top >= target.top && piece.bottom <= target.bottom) {
        state.progress = 1; finish("success", "block_inside_slot");
      } else { resetBlock(); mistake("Place the entire block inside the slot, then release."); }
    });
    block.addEventListener("pointercancel", () => {
      if (!drag) return;
      drag = null; resetBlock(); emit("drag_cancelled");
    });
  } else if (task.id === "sequence") {
    const progress = document.createElement("p");
    progress.className = "progress";
    progress.textContent = "Progress: 0 / 4";
    const grid = document.createElement("div");
    grid.className = "number-grid";
    shuffled([1, 2, 3, 4]).forEach((number) => {
      const button = document.createElement("button");
      button.className = "number";
      button.textContent = number;
      button.addEventListener("click", () => {
        if (state.status !== "in_progress") return;
        if (number !== state.progress + 1) {
          state.progress = 0;
          grid.querySelectorAll("button").forEach((item) => item.classList.remove("done"));
          mistake("Wrong order. Start again at 1.");
        } else {
          state.progress++;
          button.classList.add("done");
          emit("sequence_advanced", {progress: state.progress});
          feedback(`Correct. ${state.progress} of 4 selected.`);
          if (state.progress === 4) finish("success", "sequence_complete");
        }
        progress.textContent = `Progress: ${state.progress} / 4`;
      });
      grid.append(button);
    });
    board.append(progress, grid);
  } else if (task.id === "moving") {
    const toolbar = document.createElement("div");
    toolbar.className = "moving-toolbar";
    const start = document.createElement("button");
    start.className = "primary";
    start.textContent = "Start";
    const timer = document.createElement("span");
    timer.id = "timer";
    timer.textContent = "20.0 s · starts on Start";
    toolbar.append(start, timer);
    const arena = document.createElement("div");
    arena.className = "moving-arena";
    const dot = document.createElement("button");
    dot.className = "moving-dot";
    dot.setAttribute("aria-label", "Moving dot");
    arena.append(dot);
    board.append(toolbar, arena);
    const phase = random() * 2 * Math.PI;
    let motionStart = null;
    const position = (seconds) => {
      dot.style.left = `${(arena.clientWidth - 80) * (0.5 + 0.42 * Math.sin(phase + seconds * 0.35))}px`;
      dot.style.top = `${75 + 55 * Math.sin(phase + seconds * 0.5)}px`;
    };
    position(0);
    refreshDeadline = () => {
      if (motionStart !== null && state.status === "in_progress" && performance.now() - motionStart >= 20000) {
        timer.textContent = "0.0 s";
        finish("failure", "time_limit");
      }
    };
    const animate = () => {
      refreshDeadline();
      if (state.status !== "in_progress") return;
      const elapsed = performance.now() - motionStart;
      position(elapsed / 1000);
      timer.textContent = `${Math.max(0, (20000 - elapsed) / 1000).toFixed(1)} s · real time`;
      requestAnimationFrame(animate);
    };
    start.addEventListener("click", () => {
      if (state.status !== "in_progress" || motionStart !== null) return;
      motionStart = performance.now();
      start.disabled = true;
      feedback("Catch the dot. The clock is running.");
      emit("motion_started");
      animate();
    });
    dot.addEventListener("click", (event) => {
      refreshDeadline();
      if (state.status !== "in_progress") return;
      if (motionStart === null) { mistake("Press Start before catching the dot."); return; }
      if (event.detail === 0) { mistake("Use the mouse to catch the moving dot."); return; }
      state.progress = 1; finish("success", "dot_caught");
    });
    arena.addEventListener("click", (event) => {
      refreshDeadline();
      if (event.target !== dot && motionStart !== null) mistake("Missed the dot. Try again before time runs out.");
    });
  }
})().catch((error) => {
  const element = document.getElementById("boot-error");
  element.hidden = false;
  element.textContent = error.message;
});
