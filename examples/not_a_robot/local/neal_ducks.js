/* Nine moving entities from the captured L22 instance, not a click counter.
 * Inferred local parameters are fixed before model evaluation: a 600x590 field,
 * 100px rectangular duck hitboxes, speed 100..160px/s, reflected straight paths,
 * 160ms sprite frames, and a 300ms return to the duck's own grid cell.
 */
"use strict";

window.nealTasks[22] = {
  assets: ["level22_spritesheet.webp", "level22_net.webp"],
  render({root, asset, button, heading, newGrid, verifyFooter, state, emit, active, finish, reject, random}) {
    const width = 600;
    const height = 590;
    const size = 100;
    const returnMs = 300;
    const frameMs = 160;
    heading("Select all of the", "Ducks");
    root.style.position = "relative";
    const grid = newGrid(3);
    grid.classList.add("neal-duck-grid");
    grid.style.gap = "0";
    for (let index = 0; index < 9; index++) {
      const home = document.createElement("div");
      Object.assign(home.style, {borderRight: index % 3 < 2 ? "1px solid #ddd" : "0",
        borderBottom: index < 6 ? "1px solid #ddd" : "0", boxSizing: "border-box"});
      grid.append(home);
    }
    const field = document.createElement("div");
    field.className = "neal-duck-field";
    Object.assign(field.style, {position: "absolute", left: "0", top: "0", width: `${width}px`,
      height: `${height}px`, pointerEvents: "none", zIndex: "3"});
    root.append(field);
    const gridRect = grid.getBoundingClientRect();
    const fieldRect = field.getBoundingClientRect();
    const ducks = Array.from({length: 9}, (_, index) => {
      const x = random() * (width - size);
      const y = random() * (height - size);
      const speed = 100 + random() * 60;
      const angle = random() * Math.PI * 2;
      const node = button("neal-duck", `Duck ${index + 1}`);
      node.dataset.duck = String(index);
      node.setAttribute("aria-pressed", "false");
      Object.assign(node.style, {position: "absolute", width: `${size}px`, height: `${size}px`,
        background: "transparent", pointerEvents: "auto", padding: "0", border: "0", cursor: "none",
        userSelect: "none", touchAction: "manipulation"});
      const face = document.createElement("span");
      face.className = "neal-duck-sprite";
      Object.assign(face.style, {display: "block", width: "100%", height: "100%", pointerEvents: "none",
        backgroundImage: `url("${asset("level22_spritesheet.webp")}")`, backgroundSize: "300% 100%",
        backgroundRepeat: "no-repeat"});
      node.append(face);
      field.append(node);
      return {index, node, face, initial: {x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed},
        home: {x: gridRect.left - fieldRect.left + (index % 3 + .5) * gridRect.width / 3 - size / 2,
          y: gridRect.top - fieldRect.top + (Math.floor(index / 3) + .5) * gridRect.height / 3 - size / 2},
        phase: "roaming", x, y, returnStart: 0, returnFrom: null};
    });
    const net = document.createElement("img");
    net.className = "neal-duck-net";
    net.src = asset("level22_net.webp");
    net.alt = "";
    net.draggable = false;
    Object.assign(net.style, {position: "absolute", width: "100px", height: `${100 * 250 / 375}px`,
      pointerEvents: "none", zIndex: "10", display: "none"});
    field.append(net);
    let epoch = performance.now();
    let animationFrame = null;
    let running = false;

    const moveNet = (event) => {
      if (!running || !active()) return;
      const rect = field.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      net.style.display = x >= 0 && x <= width && y >= 0 && y <= height ? "block" : "none";
      net.style.left = `${x - 27}px`;
      net.style.top = `${y - 20}px`;
    };
    const recordMiss = (event) => {
      if (!running || !active() || event.target.closest("button")) return;
      const rect = field.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (x >= 0 && x <= width && y >= 0 && y <= height) emit("duck_missed", {x, y});
    };
    const reflected = (distance, limit) => {
      const phase = (distance % (2 * limit) + 2 * limit) % (2 * limit);
      return {position: phase <= limit ? phase : 2 * limit - phase, direction: phase <= limit ? 1 : -1};
    };
    const draw = (now) => {
      const elapsed = (now - epoch) / 1000;
      for (const duck of ducks) {
        let facingRight = false;
        if (duck.phase === "roaming") {
          const horizontal = reflected(duck.initial.x + duck.initial.vx * elapsed, width - size);
          const vertical = reflected(duck.initial.y + duck.initial.vy * elapsed, height - size);
          duck.x = horizontal.position;
          duck.y = vertical.position;
          facingRight = duck.initial.vx * horizontal.direction > 0;
        } else if (duck.phase === "returning") {
          const fraction = Math.min(1, (now - duck.returnStart) / returnMs);
          const eased = 1 - (1 - fraction) ** 3;
          duck.x = duck.returnFrom.x + (duck.home.x - duck.returnFrom.x) * eased;
          duck.y = duck.returnFrom.y + (duck.home.y - duck.returnFrom.y) * eased;
          facingRight = duck.home.x > duck.returnFrom.x;
          if (fraction === 1) {
            duck.phase = "caught";
            state.progress = ducks.filter((item) => item.phase === "caught").length;
            emit("duck_caught", {duck: duck.index, caught_count: state.progress});
          }
        }
        duck.node.dataset.phase = duck.phase;
        duck.node.setAttribute("aria-pressed", String(duck.phase === "caught"));
        duck.node.style.left = `${duck.x}px`;
        duck.node.style.top = `${duck.y}px`;
        duck.node.style.zIndex = duck.phase === "roaming" ? "3" : duck.phase === "returning" ? "2" : "1";
        const frame = duck.phase === "caught" ? 0 : Math.floor((now - epoch) / frameMs) % 3;
        duck.face.style.backgroundPosition = `${frame * 50}% 50%`;
        // The unmodified source sprite faces left. Caught ducks match that static reference pose.
        duck.face.style.transform = facingRight && duck.phase !== "caught" ? "scaleX(-1)" : "none";
      }
    };
    const tick = (now) => {
      if (!running || !active()) return;
      draw(now);
      animationFrame = requestAnimationFrame(tick);
    };
    const stopDynamic = () => {
      running = false;
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      animationFrame = null;
      window.removeEventListener("pointermove", moveNet);
      window.removeEventListener("click", recordMiss);
      window.removeEventListener("pagehide", stopDynamic);
    };
    for (const duck of ducks) {
      duck.node.addEventListener("click", () => {
        if (!active() || !running) return;
        if (duck.phase !== "roaming") {
          emit("duck_click_ignored", {duck: duck.index, phase: duck.phase});
          return;
        }
        // A real browser hit on this moving rectangle selects only this entity.
        // No nearest-target lookup, stale observation match, or DOM force-click is used.
        duck.phase = "returning";
        duck.returnStart = performance.now();
        duck.returnFrom = {x: duck.x, y: duck.y};
        duck.node.dataset.phase = "returning";
        emit("duck_return_started", {duck: duck.index, x: duck.x, y: duck.y});
      });
    }
    const refresh = () => {
      stopDynamic();
      epoch = performance.now();
      state.progress = 0;
      for (const duck of ducks) {
        duck.phase = "roaming";
        duck.returnStart = 0;
        duck.returnFrom = null;
      }
      net.style.display = "none";
      running = true;
      draw(epoch);
      emit("duck_field_reset", {field_px: [width, height], hitbox_px: size, return_ms: returnMs,
        sprite_frame_ms: frameMs, paths: ducks.map((duck) => ({duck: duck.index, ...duck.initial}))});
      window.addEventListener("pointermove", moveNet);
      window.addEventListener("click", recordMiss);
      window.addEventListener("pagehide", stopDynamic);
      animationFrame = requestAnimationFrame(tick);
    };
    verifyFooter(() => {
      draw(performance.now());
      if (ducks.every((duck) => duck.phase === "caught")) finish("all_distinct_ducks_caught");
      else reject("ducks_still_roaming_or_returning");
    });
    refresh();
    return {refresh, stopDynamic};
  },
};
