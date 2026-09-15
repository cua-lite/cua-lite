/* First-party spatial mechanic adaptations from the authored rule specifications.
 * Original procedural artwork is deliberately not presented as captured media.
 * Answers, evaluator state and opponent scores remain in closures, not snapshot.
 * Modules 1069/1087/1101/511/1092/1079 supplied factual rule constraints only.
 */
"use strict";

(() => {
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const note = (root, message) => {
    const node = document.createElement("p");
    node.className = "neal-spatial-note";
    node.textContent = message;
    root.append(node);
  };
  const canvasPanel = (root, height, label) => {
    const canvas = document.createElement("canvas");
    canvas.className = "neal-spatial-canvas";
    canvas.width = 440;
    canvas.height = height;
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", label);
    root.append(canvas);
    return canvas;
  };
  const answerField = (root) => {
    const input = document.createElement("input");
    input.className = "neal-spatial-input";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "Answer";
    input.setAttribute("aria-label", "Answer");
    root.append(input);
    return input;
  };

  window.nealTasks[16] = {
    assets: [],
    render({root, heading, verifyFooter, state, emit, active, finish, reject, random}) {
      heading("Type the characters", "Now in 3D!");
      const scene = document.createElement("div");
      scene.className = "neal-text3d";
      scene.setAttribute("aria-label", "Rotating three-dimensional letters; drag to orbit");
      const world = document.createElement("div");
      world.className = "neal-text3d-world";
      scene.append(world);
      root.append(scene);
      note(root, "Local 3D adaptation: original layered glyphs, not the original font mesh or artwork. Drag to change viewpoint.");
      const input = answerField(root);
      let answer, letters, yaw = 0, pitch = 0, dragging = null, frame, last = null;
      const colors = ["#3185cf", "#d14d37", "#45893e", "#7957b9", "#d9901a", "#179893"];
      const regenerate = () => {
        answer = Array.from({length: 6}, () => "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[Math.floor(random() * 36)]).join("");
        world.replaceChildren();
        letters = [...answer].map((character, index) => {
          const mesh = document.createElement("div");
          mesh.className = "neal-text3d-letter";
          const size = 40 + random() * 30;
          mesh.style.fontSize = `${size}px`;
          for (let depth = 0; depth < 9; depth++) {
            const layer = document.createElement("span");
            layer.textContent = character;
            layer.style.color = depth === 8 ? colors[index] : "#454754";
            layer.style.transform = `translateZ(${depth - 4}px)`;
            if (depth !== 8) layer.setAttribute("aria-hidden", "true");
            mesh.append(layer);
          }
          world.append(mesh);
          return {mesh, x: (index - 2.5) * 57, y: (random() - .5) * 36,
            z: (random() - .5) * 45, rx: random() * Math.PI, ry: random() * Math.PI,
            rz: random() * Math.PI, sx: random() * .02, sy: random() * .02, sz: random() * .05 - .015};
        });
        for (let index = 0; index < 10; index++) {
          const orb = document.createElement("span");
          orb.className = "neal-text3d-orb";
          const size = 7 + random() * 20;
          Object.assign(orb.style, {width: `${size}px`, height: `${size}px`,
            transform: `translate3d(${(random() - .5) * 380}px,${(random() - .5) * 200}px,${(random() - .5) * 100}px)`});
          world.append(orb);
        }
        input.value = "";
        state.progress = 0;
        emit("spatial_instance", {level: 16, generator: "seeded_six_alphanumeric", artwork: "original_css3d"});
      };
      const verify = verifyFooter(() => {
        if (input.value === answer) { state.progress = 1; finish("six_3d_characters_exact"); }
        else reject("3d_text_mismatch");
      });
      input.addEventListener("keydown", event => { if (event.key === "Enter") verify.click(); });
      scene.addEventListener("pointerdown", event => {
        if (!active()) return;
        dragging = [event.clientX, event.clientY];
        scene.setPointerCapture(event.pointerId);
      });
      scene.addEventListener("pointermove", event => {
        if (!dragging || !active()) return;
        yaw = clamp(yaw + (event.clientX - dragging[0]) * .01, -Math.PI / 4, Math.PI / 4);
        pitch = clamp(pitch - (event.clientY - dragging[1]) * .01, -Math.PI / 6, Math.PI / 6);
        dragging = [event.clientX, event.clientY];
        emit("view_orbit", {yaw, pitch});
      });
      scene.addEventListener("pointerup", () => { dragging = null; });
      scene.addEventListener("pointercancel", () => { dragging = null; });
      const draw = now => {
        const dt = last === null ? .016 : Math.min(.1, (now - last) / 1000);
        last = now;
        world.style.transform = `rotateX(${pitch}rad) rotateY(${yaw}rad)`;
        for (const [index, letter] of letters.entries()) {
          letter.rx += letter.sx * dt * 60;
          letter.ry += letter.sy * dt * 60;
          letter.rz += letter.sz * dt * 60;
          letter.mesh.style.transform = `translate3d(${letter.x + 2 * Math.cos(now / 500 + index)}px,${letter.y}px,${letter.z}px) rotateX(${letter.rx}rad) rotateY(${letter.ry}rad) rotateZ(${letter.rz}rad)`;
        }
        if (active()) frame = requestAnimationFrame(draw);
      };
      regenerate();
      frame = requestAnimationFrame(draw);
      return {refresh: regenerate, stopDynamic: () => cancelAnimationFrame(frame)};
    },
  };

  window.nealTasks[19] = {
    assets: [],
    render({root, heading, verifyFooter, state, emit, active, finish, reject, random}) {
      heading("Find the hidden characters", "In the Dark");
      const canvas = canvasPanel(root, 350, "Dark wall; move the pointer to illuminate letters");
      const graphics = canvas.getContext("2d");
      note(root, "Original procedural dark scene. Flashlight and flicker are functional; original horror font, video and audio are not bundled.");
      const input = answerField(root);
      let characters, answer, light = [-100, -100], frame;
      const start = performance.now();
      const regenerate = () => {
        characters = Array.from({length: 5}, (_, index) => ({
          character: "ACEFGHIJKLMNPRSUVWXYZ12456789"[Math.floor(random() * 29)],
          x: (index * 20 + 7) * 4.4, y: (10 + Math.floor(random() * 75)) * 3.5,
          size: 43 + Math.floor(random() * 29),
        }));
        answer = characters.map(row => row.character).join("");
        input.value = "";
        state.progress = 0;
        emit("spatial_instance", {level: 19, generator: "seeded_five_flashlight_characters", artwork: "original_canvas"});
      };
      canvas.addEventListener("pointermove", event => {
        const box = canvas.getBoundingClientRect();
        light = [(event.clientX - box.left) * 440 / box.width, (event.clientY - box.top) * 350 / box.height];
      });
      const verify = verifyFooter(() => {
        if (input.value.toLowerCase() === answer.toLowerCase()) { state.progress = 1; finish("flashlight_characters_match"); }
        else reject("flashlight_text_mismatch");
      });
      input.addEventListener("keydown", event => { if (event.key === "Enter") verify.click(); });
      const draw = now => {
        graphics.fillStyle = "#323032";
        graphics.fillRect(0, 0, 440, 350);
        graphics.strokeStyle = "#777";
        for (let y = 0; y < 350; y += 35) {
          graphics.beginPath(); graphics.moveTo(0, y); graphics.lineTo(440, y); graphics.stroke();
          for (let x = (y % 70 ? 30 : 0); x < 440; x += 80) graphics.strokeRect(x, y, 80, 35);
        }
        for (const row of characters) {
          graphics.font = `bold ${row.size}px Georgia,serif`;
          graphics.fillStyle = "#bf302b";
          graphics.textBaseline = "middle";
          graphics.fillText(row.character, row.x, row.y);
        }
        const phase = (now - start) % 3000;
        const flicker = phase >= 2640 && phase < 2760 || phase >= 2880;
        const radius = flicker ? 28 : 115;
        const beam = graphics.createRadialGradient(...light, 0, ...light, radius);
        beam.addColorStop(0, flicker ? "rgba(0,0,0,.8)" : "rgba(0,0,0,.1)");
        beam.addColorStop(1, "#000");
        graphics.fillStyle = beam;
        graphics.fillRect(0, 0, 440, 350);
        if (active()) frame = requestAnimationFrame(draw);
      };
      regenerate(); frame = requestAnimationFrame(draw);
      return {refresh: regenerate, stopDynamic: () => cancelAnimationFrame(frame)};
    },
  };

  window.nealTasks[23] = {
    assets: [],
    render({root, heading, button, verifyFooter, state, emit, active, finish, reject, random}) {
      const challenges = [
        {name: "Couple Kissing", yaw: [-171, -143], pitch: [-35, -14], fov: 40},
        {name: "Guitar Cat", yaw: [71, 87], pitch: [-38, -27], fov: 16},
        {name: "Chilli's Sign", yaw: [68, 76], pitch: [-11, -2], fov: 12},
      ];
      let index = Math.floor(random() * 2), yaw = 0, pitch = 0, fov = 100, dragging;
      heading("Pan and zoom to frame", challenges[index].name);
      const title = root.querySelector(".neal-heading strong");
      const canvas = canvasPanel(root, 340, "Panoramic scene; drag to look around and scroll to zoom");
      const graphics = canvas.getContext("2d");
      note(root, "Original spherical illustrated panorama, not the original photograph. Target direction and field-of-view rules are source-derived.");
      const controls = document.createElement("div"); controls.className = "neal-spatial-controls";
      for (const [label, change] of [["Zoom in", -.2], ["Zoom out", .2]]) {
        const node = button("", label); node.textContent = label;
        node.addEventListener("click", () => { if (active()) { fov = clamp(fov * (1 + change), 10, 100); draw(); } });
        controls.append(node);
      }
      root.append(controls);
      const project = (longitude, latitude) => {
        const a = (longitude - yaw) * Math.PI / 180, b = latitude * Math.PI / 180, p = pitch * Math.PI / 180;
        const x = Math.cos(b) * Math.sin(a), y = Math.sin(b), z = Math.cos(b) * Math.cos(a);
        const cy = y * Math.cos(p) - z * Math.sin(p), cz = y * Math.sin(p) + z * Math.cos(p);
        if (cz <= .02) return null;
        const focal = 220 / Math.tan(fov * Math.PI / 360);
        return [220 + focal * x / cz, 170 - focal * cy / cz, focal / cz];
      };
      const draw = () => {
        const sky = graphics.createLinearGradient(0, 0, 0, 340);
        sky.addColorStop(0, "#94c7e6"); sky.addColorStop(1, "#e7d6bd");
        graphics.fillStyle = sky; graphics.fillRect(0, 0, 440, 340);
        // Reproject an authored spherical cityscape rather than sliding a flat image.
        for (let longitude = -180; longitude < 180; longitude += 15) {
          const top = project(longitude, 10), bottom = project(longitude, -40);
          if (!top || !bottom) continue;
          const width = top[2] * .18;
          graphics.fillStyle = longitude % 30 ? "#998e7e" : "#b6a490";
          graphics.fillRect(top[0] - width / 2, top[1], width, bottom[1] - top[1]);
          graphics.fillStyle = "#526371";
          for (let floor = -30; floor < 10; floor += 8) {
            const windowPoint = project(longitude, floor);
            if (windowPoint) graphics.fillRect(windowPoint[0] - width / 5, windowPoint[1], width / 3, width / 4);
          }
        }
        challenges.forEach((row, item) => {
          const point = project((row.yaw[0] + row.yaw[1]) / 2, (row.pitch[0] + row.pitch[1]) / 2);
          if (!point) return;
          const [x, y, scale] = point, s = scale * (item === 0 ? .17 : item === 1 ? .065 : .04);
          graphics.save(); graphics.translate(x, y);
          if (item === 0) {
            for (const direction of [-1, 1]) {
              graphics.fillStyle = direction < 0 ? "#b44661" : "#284963";
              graphics.fillRect(direction * s * .4 - s * .22, 0, s * .44, s);
              graphics.fillStyle = "#e7af83"; graphics.beginPath(); graphics.arc(direction * s * .22, -s * .2, s * .29, 0, Math.PI * 2); graphics.fill();
            }
            graphics.fillStyle = "#d82956"; graphics.font = `${s * .5}px serif`; graphics.fillText("♥", -s * .2, -s * .65);
          } else if (item === 1) {
            graphics.fillStyle = "#d78837"; graphics.beginPath(); graphics.arc(0, 0, s, 0, Math.PI * 2); graphics.fill();
            graphics.beginPath(); graphics.moveTo(-s, -.5 * s); graphics.lineTo(-s, -1.5 * s); graphics.lineTo(0, -.8 * s); graphics.fill();
            graphics.beginPath(); graphics.moveTo(s, -.5 * s); graphics.lineTo(s, -1.5 * s); graphics.lineTo(0, -.8 * s); graphics.fill();
            graphics.fillStyle = "#222"; graphics.fillRect(-s * .5, -s * .2, s * .2, s * .2); graphics.fillRect(s * .3, -s * .2, s * .2, s * .2);
            graphics.fillStyle = "#7c451b"; graphics.beginPath(); graphics.ellipse(0, s, s * .7, s, -.4, 0, Math.PI * 2); graphics.fill();
            graphics.fillRect(0, -.3 * s, s * .2, s * 1.5);
          } else {
            graphics.fillStyle = "#304733"; graphics.fillRect(-s * 2, -s, s * 4, s * 2);
            graphics.fillStyle = "#fff"; graphics.textAlign = "center"; graphics.font = `bold ${s * .75}px sans-serif`; graphics.fillText("CHILLI'S", 0, s * .3);
          }
          graphics.restore();
        });
        graphics.strokeStyle = "#fff9"; graphics.beginPath(); graphics.moveTo(208, 170); graphics.lineTo(232, 170); graphics.moveTo(220, 158); graphics.lineTo(220, 182); graphics.stroke();
      };
      canvas.addEventListener("pointerdown", event => { if (active()) { dragging = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); } });
      canvas.addEventListener("pointermove", event => {
        if (!dragging || !active()) return;
        const width = canvas.getBoundingClientRect().width;
        yaw = ((yaw - (event.clientX - dragging[0]) * fov / width + 540) % 360) - 180;
        pitch = clamp(pitch + (event.clientY - dragging[1]) * fov / width, -85, 85);
        dragging = [event.clientX, event.clientY]; draw();
      });
      const release = () => { if (dragging) emit("panorama_view", {yaw, pitch, hfov: fov}); dragging = null; };
      canvas.addEventListener("pointerup", release); canvas.addEventListener("pointercancel", release);
      canvas.addEventListener("wheel", event => { event.preventDefault(); if (active()) { fov = clamp(fov * Math.exp(event.deltaY * .001), 10, 100); draw(); } }, {passive: false});
      verifyFooter(() => {
        const row = challenges[index];
        emit("panorama_submit_view", {yaw, pitch, hfov: fov, target: row.name});
        if (yaw >= row.yaw[0] && yaw <= row.yaw[1] && pitch >= row.pitch[0] && pitch <= row.pitch[1] && fov <= row.fov) {
          state.progress = 1; finish("panorama_target_framed");
        } else reject("panorama_view_outside_target_bounds");
      });
      draw();
      return {refresh: () => { index = (index + 1) % 3; yaw = 0; pitch = 0; fov = 100; dragging = null; title.textContent = challenges[index].name; draw(); }};
    },
  };

  window.nealTasks[38] = {
    assets: [],
    render({root, heading, button, verifyFooter, state, emit, active, finish, reject}) {
      heading("Park without hitting anyone", "Tough Decisions");
      const canvas = canvasPanel(root, 440, "Parking simulation; hold arrow keys or WASD to drive");
      const graphics = canvas.getContext("2d");
      note(root, "Original schematic artwork with real car motion and collisions. Fictional simulation; original sound and photo textures are not bundled. The road continues outside the view.");
      const status = document.createElement("div"); status.className = "neal-spatial-status"; root.append(status);
      const obstacles = [[95, 5, 58, 100], [25, 5, 58, 100], [240, 5, 58, 100], [310, 5, 58, 100], [380, 5, 58, 100], [18, 187, 85, 40], [300, 187, 85, 40], [181, 192, 35, 35]];
      const people = [[230, 182, 52, 52], [125, 185, 45, 45]];
      let car, hit, lost, keys = new Set(), frame, last = null, resetAt = null;
      const reset = () => {
        car = {x: 155, y: 355, angle: -Math.PI / 2, speed: 0, steering: 0, accelerationTime: 0, momentum: 0};
        hit = [false, false]; lost = false; resetAt = null; keys.clear(); state.progress = 0; status.textContent = "Hold ↑ to accelerate, ↓ to reverse; steer with ← →.";
        emit("parking_reset");
      };
      const normalizedKey = key => ({arrowup: "w", arrowdown: "s", arrowleft: "a", arrowright: "d"}[key.toLowerCase()] || key.toLowerCase());
      const keydown = event => { const key = normalizedKey(event.key); if ("wasd".includes(key) && key.length === 1 && active()) { event.preventDefault(); keys.add(key); } };
      const keyup = event => { keys.delete(normalizedKey(event.key)); };
      const blur = () => keys.clear();
      window.addEventListener("keydown", keydown); window.addEventListener("keyup", keyup); window.addEventListener("blur", blur);
      const controls = document.createElement("div"); controls.className = "neal-spatial-controls";
      for (const [key, label] of [["a", "Steer left"], ["w", "Accelerate"], ["s", "Reverse"], ["d", "Steer right"]]) {
        const node = button("", label); node.textContent = {a: "←", w: "↑", s: "↓", d: "→"}[key];
        node.addEventListener("pointerdown", event => { if (active()) { keys.add(key); node.setPointerCapture(event.pointerId); } });
        node.addEventListener("pointerup", () => keys.delete(key)); node.addEventListener("pointercancel", () => keys.delete(key)); controls.append(node);
      }
      root.append(controls);
      const collides = rectangle => {
        const [x, y, width, height] = rectangle, cx = car.x + 40, cy = car.y + 25;
        const c = Math.cos(car.angle), s = Math.sin(car.angle);
        const corners = [[-32, -20], [32, -20], [32, 20], [-32, 20]].map(([a, b]) => [cx + a * c - b * s, cy + a * s + b * c]);
        if (corners.some(([a, b]) => a >= x && a <= x + width && b >= y && b <= y + height)) return true;
        return [[x, y], [x + width, y], [x, y + height], [x + width, y + height]].some(([a, b]) => {
          const dx = a - cx, dy = b - cy;
          return Math.abs(dx * c + dy * s) <= 32 && Math.abs(-dx * s + dy * c) <= 20;
        });
      };
      const integrate = dt => {
        const steering = keys.has("a") ? -1 : keys.has("d") ? 1 : 0;
        car.steering = steering ? clamp(car.steering + steering * .05 * dt, -Math.PI / 4, Math.PI / 4) : Math.sign(car.steering) * Math.max(0, Math.abs(car.steering) - .05 * dt);
        const throttle = keys.has("w") ? 1 : keys.has("s") ? -1 : 0;
        if (throttle) {
          car.accelerationTime = Math.min(120, car.accelerationTime + dt);
          car.speed += throttle * .02 * (1 + 3 * car.accelerationTime / 120) * dt;
          car.momentum = Math.min(.9, car.momentum + .15 * dt);
        } else {
          car.accelerationTime = 0;
          car.speed *= (.88 + .1 * car.momentum) ** dt;
          car.momentum *= .97 ** dt;
        }
        car.speed = clamp(car.speed * .985 ** dt, -2, 2);
        if (Math.abs(car.speed) < .005) car.speed = 0;
        if (Math.abs(car.speed) > .1) car.angle += car.steering * car.speed * .03 * dt;
        car.x += Math.cos(car.angle) * car.speed * dt; car.y += Math.sin(car.angle) * car.speed * dt;
        if (obstacles.some(collides)) { lost = true; resetAt = performance.now() + 2000; reject("parking_obstacle_collision"); status.textContent = "Collision. Resetting in two seconds."; }
        if (!lost) people.forEach((person, index) => { if (!hit[index] && collides(person)) { hit[index] = true; emit("parking_person_hit", {person: index}); status.textContent = "A person was hit. This attempt cannot pass."; } });
      };
      const draw = now => {
        // Cap/substep delayed local frames to avoid tunnelling through obstacles.
        // This is an explicit local robustness difference from uncapped source dt.
        let dt = last === null ? 1 : Math.min(6, (now - last) / 16.67); last = now;
        if (resetAt && now >= resetAt) reset();
        while (dt > 0 && !lost) { const step = Math.min(dt, 1); integrate(step); dt -= step; }
        graphics.setTransform(1.1, 0, 0, 1.1, 0, 0);
        graphics.fillStyle = "#d7dbd7"; graphics.fillRect(0, 0, 400, 400);
        graphics.fillStyle = "#65aa70"; graphics.fillRect(155, 4, 85, 105);
        graphics.strokeStyle = "white"; graphics.lineWidth = 3; graphics.strokeRect(155, 4, 85, 105);
        graphics.fillStyle = "#fff"; graphics.font = "bold 34px sans-serif"; graphics.fillText("P", 185, 58);
        for (const [x, y, w, h] of obstacles) { graphics.fillStyle = "#737b87"; graphics.fillRect(x, y, w, h); graphics.strokeStyle = "#454b55"; graphics.strokeRect(x, y, w, h); }
        people.forEach(([x, y, w, h], index) => {
          graphics.fillStyle = hit[index] ? "#ad3945" : "#e6b968";
          graphics.beginPath(); graphics.arc(x + w / 2, y + h / 2, w / 2, 0, Math.PI * 2); graphics.fill();
          graphics.fillStyle = "#222"; graphics.font = "12px sans-serif"; graphics.fillText(index ? "BABY" : "PERSON", x + 3, y + h / 2 + 4);
        });
        graphics.save(); graphics.translate(car.x + 40, car.y + 25); graphics.rotate(car.angle);
        graphics.fillStyle = lost ? "#a54743" : "#f4f8fb"; graphics.fillRect(-40, -25, 80, 50);
        graphics.strokeStyle = "#273d51"; graphics.strokeRect(-40, -25, 80, 50);
        graphics.fillStyle = "#578fae"; graphics.fillRect(9, -20, 15, 40);
        graphics.fillStyle = "#f0d442"; graphics.fillRect(35, -19, 5, 10); graphics.fillRect(35, 9, 5, 10);
        graphics.restore();
        graphics.setTransform(1, 0, 0, 1, 0, 0);
        if (active()) frame = requestAnimationFrame(draw);
      };
      verifyFooter(() => {
        emit("parking_submit_pose", {x: car.x, y: car.y, angle: car.angle, speed: car.speed, hit: [...hit], lost});
        if (hit.some(Boolean)) { reject("parking_person_hit"); reset(); }
        else if (!lost && car.x >= 150 && car.x <= 165 && car.y >= -1 && car.y <= 64) { state.progress = 1; finish("parked_without_hitting_people"); }
        else reject("car_outside_parking_space");
      });
      reset(); frame = requestAnimationFrame(draw);
      return {refresh: reset, stopDynamic: () => { cancelAnimationFrame(frame); keys.clear(); window.removeEventListener("keydown", keydown); window.removeEventListener("keyup", keyup); window.removeEventListener("blur", blur); }};
    },
  };

  window.nealTasks[43] = {
    assets: [],
    render({root, heading, verifyFooter, state, emit, active, finish, reject}) {
      heading("Assemble the chair", "Ikea");
      const svgNS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("viewBox", "0 0 440 440"); svg.classList.add("neal-spatial-assembly"); svg.setAttribute("aria-label", "3D chair parts; drag a part to connect, drag blank space to orbit");
      root.append(svg);
      note(root, "Original 3D geometry rendering and wood colors, not captured textures. Dots show physical attachment points. Drag parts; drag empty space to orbit.");
      let parts, connections = new Set(), yaw, pitch, drag;
      const requirements = [1, 2, 3, 4].map(number => `base-corner${number}-leg-top`).concat(["plank-connect-rod-halfway", "plank-connect-rod-top"]);
      const camera = point => {
        const [x, y, z] = point, cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
        const rx = x * cy - z * sy, rz = x * sy + z * cy;
        return [rx, y * cp - rz * sp, y * sp + rz * cp];
      };
      const project = point => { const [x, y, z] = camera(point), scale = 310 / (7 - z); return [220 + x * scale, 218 - y * scale, z]; };
      const worldPoint = (part, point) => point.offset.map((v, i) => v + part.position[i]);
      const allPoints = () => parts.flatMap(part => part.points.map(point => ({part, point, world: worldPoint(part, point)})));
      const connectionKey = (first, second) => [`${first.part.type}-${first.point.name}`, `${second.part.type}-${second.point.name}`].sort().join("-");
      const recalculate = () => {
        const points = allPoints(); connections = new Set();
        const cssScale = svg.getBoundingClientRect().width / 440;
        points.forEach((first, index) => points.slice(index + 1).forEach(second => {
          const a = project(first.world), b = project(second.world);
          if (Math.hypot(a[0] - b[0], a[1] - b[1]) * cssScale <= 20) connections.add(connectionKey(first, second));
        }));
        state.progress = requirements.filter(key => connections.has(key)).length;
        emit("assembly_connections", {keys: [...connections].sort(), matched: state.progress, required: 6});
      };
      const draw = () => {
        svg.replaceChildren();
        const ordered = [...parts].sort((a, b) => camera(a.position)[2] - camera(b.position)[2]);
        for (const part of ordered) {
          const group = document.createElementNS(svgNS, "g"); group.dataset.piece = part.id;
          group.setAttribute("aria-label", `${part.type} ${part.id}`);
          const [w, h, d] = part.size;
          const vertices = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(([x,y,z]) => [part.position[0]+x*w/2,part.position[1]+y*h/2,part.position[2]+z*d/2]);
          const faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]];
          faces.sort((a,b) => a.reduce((sum,i)=>sum+camera(vertices[i])[2],0)-b.reduce((sum,i)=>sum+camera(vertices[i])[2],0));
          for (const [index, face] of faces.entries()) {
            const polygon = document.createElementNS(svgNS, "polygon");
            polygon.setAttribute("points", face.map(i=>project(vertices[i]).slice(0,2).join(",")).join(" "));
            polygon.setAttribute("fill", ["#b28555", "#d9b785", "#a87849", "#e2c49a", "#bc9160", "#caa270"][index]);
            polygon.setAttribute("stroke", "#795b3b"); polygon.setAttribute("stroke-width", ".8"); group.append(polygon);
          }
          for (const point of part.points) {
            const [x,y] = project(worldPoint(part,point)), dot = document.createElementNS(svgNS,"circle");
            dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("r", "3"); dot.dataset.connector = point.name; group.append(dot);
          }
          svg.append(group);
        }
      };
      const reset = () => {
        const endpoint = (name, offset) => ({name, offset});
        parts = [{id:"seat",type:"base",size:[2.5,.2,2.5],position:[0,0,0],points:[endpoint("corner1",[-1,-.1,-1]),endpoint("corner2",[1,-.1,-1]),endpoint("corner3",[-1,-.1,1]),endpoint("corner4",[1,-.1,1])]}];
        for(let i=0;i<4;i++) parts.push({id:`leg${i+1}`,type:"leg",size:[.4,2.5,.4],position:[.5*(i-i%2)-3,i%2*2.8-1.5,0],points:[endpoint("top",[0,1.25,0]),endpoint("bottom",[0,-1.25,0])]});
        for(let i=0;i<2;i++) parts.push({id:`rod${i+1}`,type:"rod",size:[.4,2.5,.4],position:[i+2.5,1.25,0],points:[endpoint("top",[0,1.25,0]),endpoint("bottom",[0,-1.25,0]),endpoint("halfway",[0,0,0])]});
        for(let i=0;i<2;i++) parts.push({id:`plank${i+1}`,type:"plank",size:[2.35,.2,.2],position:[0,i-2.5,0],points:[endpoint("connect",[-1,0,0]),endpoint("connect",[1,0,0])]});
        yaw=.54; pitch=.33; connections.clear(); state.progress=0; drag=null; draw();
      };
      const screen = event => { const box=svg.getBoundingClientRect(); return [(event.clientX-box.left)*440/box.width,(event.clientY-box.top)*440/box.height]; };
      svg.addEventListener("pointerdown", event => {
        if(!active()) return;
        const id=event.target.closest("[data-piece]")?.dataset.piece;
        const part=parts.find(part=>part.id===id), start=screen(event);
        drag={part,last:start,start,position:part?[...part.position]:null}; svg.setPointerCapture(event.pointerId);
      });
      svg.addEventListener("pointermove", event => {
        if(!drag || !active()) return;
        const point=screen(event), dx=point[0]-drag.last[0], dy=point[1]-drag.last[1]; drag.last=point;
        if(!drag.part) {yaw+=dx*.01;pitch=clamp(pitch+dy*.01,-1.2,1.2);draw();return;}
        const part=drag.part, depth=camera(drag.position)[2], scale=(7-depth)/310;
        // Camera-facing drag plane, transformed back into world coordinates.
        const x=(point[0]-drag.start[0])*scale, y=-(point[1]-drag.start[1])*scale, z=0;
        const ry=y*Math.cos(pitch)+z*Math.sin(pitch), rz=-y*Math.sin(pitch)+z*Math.cos(pitch);
        part.position=[drag.position[0]+x*Math.cos(yaw)+rz*Math.sin(yaw),drag.position[1]+ry,drag.position[2]-x*Math.sin(yaw)+rz*Math.cos(yaw)];
        let closest=null, distance=20;
        const cssScale=svg.getBoundingClientRect().width/440;
        for(const own of part.points) for(const other of allPoints()) {
          if(other.part===part) continue;
          const a=project(worldPoint(part,own)),b=project(other.world),d=Math.hypot(a[0]-b[0],a[1]-b[1])*cssScale;
          if(d<distance){distance=d;closest={own,other};}
        }
        if(closest) part.position=closest.other.world.map((value,i)=>value-closest.own.offset[i]);
        draw();
      });
      const release=()=>{if(drag){recalculate();drag=null;draw();}};
      svg.addEventListener("pointerup",release);svg.addEventListener("pointercancel",release);
      verifyFooter(()=>{if(requirements.every(key=>connections.has(key)))finish("chair_required_connections_present");else reject("chair_connections_incomplete");});
      reset();
      return {refresh:reset};
    },
  };

  window.nealTasks[44] = {
    assets: [],
    async render({root, heading, button, verifyFooter, state, emit, active, finish, reject}) {
      const {Chess} = await import("/vendor/spatial/chess-1.4.0.js");
      heading("Win the chess game", "Grandmaster");
      const board=document.createElement("div");board.className="neal-spatial-chess";root.append(board);
      const status=document.createElement("div");status.className="neal-spatial-status";root.append(status);
      status.setAttribute("role","status");
      note(root,"Opponent: official Stockfish 17 Lite WASM, Skill Level 10, one second per move. Local chess.js 1.4.0 rules differ from the unconfirmed original library version. Play White. Click two squares or drag; promotion is to queen. Refresh after a loss, stalemate, or ten White moves earns a replacement queen.");
      let game,selected=null,whiteMoves=0,queens=0,generation=0,winner="";
      let engine=null,enginePhase="initializing",engineError="",engineTimer=null,pendingSearch=null,closed=false;
      const symbols={p:"♟",n:"♞",b:"♝",r:"♜",q:"♛",k:"♚"};
      const squares=[];
      const canPlayWhite=()=>active()&&!closed&&enginePhase==="ready"&&!game.isGameOver()&&game.turn()==="w";
      const draw=()=>{
        winner=game.isCheckmate()?(game.turn()==="b"?"w":"b"):game.isStalemate()?"stalemate":"";
        for(const [square,node] of squares){const piece=game.get(square);node.replaceChildren();node.dataset.color=piece?.color||"";
          const mark=document.createElement("span");mark.textContent=piece?symbols[piece.type]:"";node.append(mark);
          const label=document.createElement("span");label.className="coordinate";label.textContent=square;node.append(label);
          node.classList.toggle("selected",selected===square);node.setAttribute("aria-label",`${square}${piece?` ${piece.color==="w"?"White":"Black"} ${{p:"pawn",n:"knight",b:"bishop",r:"rook",q:"queen",k:"king"}[piece.type]}`:" empty"}`);
          node.disabled=!canPlayWhite();
        }
        status.textContent=winner==="w"?"White wins by checkmate. Verify to finish."
          :enginePhase==="error"?`Engine error (${engineError}). Reload this task to retry.`
          :winner==="b"?"Black wins by checkmate. Refresh to retry."
          :winner==="stalemate"?"Draw (stalemate). Refresh to retry."
          :game.isGameOver()?"Game ended in a draw. Refresh to retry."
          :enginePhase==="stopping"?"Stopping the previous search…"
          :enginePhase==="searching"?"Black is thinking…"
          :enginePhase!=="ready"?"Loading Stockfish 17 Lite WASM…":"White to move.";
        state.progress=winner==="w"?1:0;
      };
      const failEngine=(stage,message)=>{
        if(closed||enginePhase==="error")return;
        clearTimeout(engineTimer);engineTimer=null;pendingSearch=null;
        enginePhase="error";engineError=stage;selected=null;
        // A completed legal checkmate no longer depends on an engine reply.
        if(winner!=="w"){
          state.status="infra_error";state.reason=`chess_engine_${stage}`;
          root.dataset.status=state.status;
        }
        emit("chess_engine_error",{stage,message});
        if(engine){engine.terminate();engine=null;}
        draw();
      };
      const sendEngine=line=>{
        try{engine.postMessage(line);emit("chess_engine_uci",{direction:"sent",line});return true;}
        catch(error){failEngine("transport",String(error));return false;}
      };
      const reply=()=>{
        pendingSearch={generation,fen:game.fen()};enginePhase="searching";draw();
        engineTimer=setTimeout(()=>failEngine("search_timeout","No bestmove within 10000 ms"),10000);
        if(sendEngine(`position fen ${pendingSearch.fen}`))sendEngine("go movetime 1000");
      };
      const moveWhite=(from,to)=>{
        if(!canPlayWhite())return;
        let move;
        try{move=game.move({from,to,promotion:"q"});}catch{reject("illegal_chess_move");selected=null;draw();return;}
        whiteMoves++;selected=null;emit("chess_move",{side:"w",from:move.from,to:move.to,san:move.san});reply();
      };
      for(let rank=8;rank>=1;rank--)for(const file of "abcdefgh"){
        const square=`${file}${rank}`,node=button("neal-spatial-square",square);
        if((file.charCodeAt(0)-97+rank)%2===1)node.classList.add("dark");node.dataset.square=square;
        node.addEventListener("click",()=>{if(!canPlayWhite())return;const piece=game.get(square);if(piece?.color==="w"){selected=square;draw();}else if(selected)moveWhite(selected,square);});
        node.addEventListener("pointerdown",event=>{if(!canPlayWhite()||game.get(square)?.color!=="w")return;node.setPointerCapture(event.pointerId);node.dataset.dragStart=`${event.clientX},${event.clientY}`;});
        node.addEventListener("pointerup",event=>{const start=node.dataset.dragStart;if(!start)return;delete node.dataset.dragStart;const [x,y]=start.split(",").map(Number);if(Math.hypot(event.clientX-x,event.clientY-y)<6)return;const destination=document.elementFromPoint(event.clientX,event.clientY)?.closest("[data-square]")?.dataset.square;if(destination&&destination!==square)moveWhite(square,destination);});
        node.addEventListener("pointercancel",()=>{delete node.dataset.dragStart;});
        board.append(node);squares.push([square,node]);
      }
      const reset=()=>{
        if(closed)return;
        generation++;
        if((winner==="b"||winner==="stalemate"||whiteMoves>=10)&&queens<6)queens++;
        game=new Chess();for(const square of ["a2","h2","b2","g2","d2","e2"].slice(0,queens)){game.remove(square);game.put({type:"q",color:"w"},square);}
        whiteMoves=0;winner="";selected=null;state.progress=0;
        for(const [,node] of squares)delete node.dataset.dragStart;
        emit("chess_reset",{assistance_queens:queens,generation});
        // UCI has no search IDs. Drain the old bestmove before accepting new moves.
        if(enginePhase==="searching"){
          enginePhase="stopping";clearTimeout(engineTimer);
          engineTimer=setTimeout(()=>failEngine("stop_timeout","No bestmove after stop within 5000 ms"),5000);
          sendEngine("stop");
        }
        draw();
      };
      verifyFooter(()=>{
        if(winner==="w")finish("white_wins_legal_chess_game");
        else if(enginePhase==="error")emit("chess_verify_blocked",{reason:"engine_error"});
        else reject("white_has_not_won_chess");
      });
      reset();
      try{
        engine=new Worker("/vendor/spatial/stockfish-17-lite-single.js");
        engineTimer=setTimeout(()=>failEngine("initialization_timeout","No readyok within 20000 ms"),20000);
        engine.onmessage=event=>{
          if(closed||!active()||enginePhase==="error")return;
          const line=event.data;
          if(typeof line!=="string"){failEngine("protocol","Worker returned a non-text UCI message");return;}
          emit("chess_engine_uci",{direction:"received",line});
          if(line==="uciok"&&enginePhase==="initializing"){
            enginePhase="awaiting_ready";
            if(sendEngine("setoption name Skill Level value 10"))sendEngine("isready");
          }else if(line==="readyok"&&enginePhase==="awaiting_ready"){
            clearTimeout(engineTimer);engineTimer=null;enginePhase="ready";
            emit("chess_engine_ready",{engine:"Stockfish 17 Lite WASM",skill_level:10,move_time_ms:1000});draw();
          }else if(line.startsWith("bestmove")){
            if(!pendingSearch){failEngine("protocol","Unexpected bestmove without a pending search");return;}
            clearTimeout(engineTimer);engineTimer=null;
            const search=pendingSearch;pendingSearch=null;
            if(search.generation!==generation){
              enginePhase="ready";
              emit("chess_engine_stale_result",{line,search_generation:search.generation,current_generation:generation});draw();return;
            }
            const match=/^bestmove ([a-h][1-8][a-h][1-8][qrbn]?|\(none\)|0000)(?:\s|$)/.exec(line);
            if(!match){failEngine("protocol","Malformed bestmove response");return;}
            const uci=match[1];
            if(uci==="(none)"||uci==="0000"){
              if(!game.isGameOver()){failEngine("invalid_move","Engine returned no move for an unfinished game");return;}
            }else{
              let move;
              try{move=game.move({from:uci.slice(0,2),to:uci.slice(2,4),...(uci.length===5?{promotion:uci[4]}:{})});}
              catch(error){failEngine("invalid_move",String(error));return;}
              emit("chess_move",{side:"b",from:move.from,to:move.to,san:move.san,opponent:"stockfish_17_lite_wasm"});
            }
            enginePhase="ready";selected=null;draw();
          }
        };
        engine.onerror=event=>{event.preventDefault();failEngine("worker",event.message||"Worker could not load or execute");};
        engine.onmessageerror=()=>failEngine("transport","Worker message could not be decoded");
        sendEngine("uci");
      }catch(error){failEngine("initialization",String(error));}
      return{refresh:reset,stopDynamic:()=>{
        if(closed)return;
        closed=true;generation++;clearTimeout(engineTimer);engineTimer=null;pendingSearch=null;
        if(engine){engine.terminate();engine=null;}
        enginePhase="closed";emit("chess_engine_closed");
      }};
    },
  };
})();
