"use strict";

// Authored geometric reconstructions, not copies of the reference artwork.
// Rules follow the static level specifications; all evaluation stays local.
(() => {
  const element = (tag, parent, style = {}, text = "") => {
    const node = document.createElement(tag);
    Object.assign(node.style, style);
    node.textContent = text;
    if (parent) parent.append(node);
    return node;
  };
  const label = (parent, name, text) => {
    const node = element("div", parent, {padding: "8px", textAlign: "center"}, text);
    node.className = name;
    node.setAttribute("role", "status");
    return node;
  };
  const action = (ctx, parent, name, text = name) => {
    const node = ctx.button("neal-motion-control", name);
    node.textContent = text;
    Object.assign(node.style, {padding: "8px 12px", border: "1px solid #aaa", cursor: "pointer"});
    parent.append(node);
    return node;
  };
  const canvas = (parent, size = 400) => {
    const node = element("canvas", parent, {display: "block", width: "100%", aspectRatio: "1", touchAction: "none"});
    node.width = node.height = size;
    return node;
  };
  const point = (event, node, size) => {
    const rect = node.getBoundingClientRect();
    return [(event.clientX - rect.left) * size / rect.width, (event.clientY - rect.top) * size / rect.height];
  };
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  // Parking coordinates are geometric rule data in the source specification.
  const parkingStages = [
    {start: [160, 300, -Math.PI / 2], slot: [300, 107, 90, 50],
      obstacles: [[20,50,90,50],[20,110,80,50],[20,170,80,50],[20,230,80,50],[20,290,87,50],[20,350,80,50],[300,50,80,40],[300,170,85,45],[300,230,85,45],[300,290,85,45],[300,350,85,45],[0,18,380,2]], moving: []},
    {start: [10, 50, 0], slot: [280, 300, 90, 50],
      obstacles: [[265,365,90,50],[293,240,90,42],[310,180,90,42],[320,117,90,42],[330,58,90,42],[347,0,90,34],[0,0,110,34],[0,115,90,34],[0,145,80,34],[0,175,70,34],[0,205,65,34],[0,235,55,34],[0,265,45,34],[0,295,40,34],[0,325,35,34],[0,355,30,50],[375,300,30,50],[370,340,30,50]],
      moving: [[120,450,54,92,Math.PI/12],[160,-100,54,92,Math.PI/12+Math.PI]]},
    {start: [260, 350, -Math.PI / 2], slot: [80, 5, 100, 55],
      obstacles: [[0,10,64,40],[196,10,90,40],[310,6,90,50],[0,360,200,30],[196,370,20,30],[220,380,10,30],[370,290,30,110],[380,280,30,10],[390,260,30,15],[0,0,250,1],[0,0,1,350],[398,0,1,350]], moving: []},
    {start: [10, 300, 0], slot: [80, 5, 100, 55],
      obstacles: [[0,10,64,40],[196,10,90,40],[310,6,90,50],[0,374,400,30],[0,279,289,3],[398,1,3,400],[1,1,2,280],[80,1,100,2]],
      moving: [[450,80,54,92,-Math.PI/2],[-100,180,54,92,Math.PI/2]]},
  ];
  const reducedCar = car => ({x: car.x + .1 * car.w, y: car.y + .1 * car.h, w: .8 * car.w, h: .8 * car.h, a: car.a});
  const corners = rect => [[-1,-1],[1,-1],[1,1],[-1,1]].map(([dx, dy]) => {
    const x = dx * rect.w / 2, y = dy * rect.h / 2;
    return [rect.x + rect.w / 2 + x * Math.cos(rect.a) - y * Math.sin(rect.a),
      rect.y + rect.h / 2 + x * Math.sin(rect.a) + y * Math.cos(rect.a)];
  });
  const contains = (rect, [x, y]) => {
    const dx = x - rect.x - rect.w / 2, dy = y - rect.y - rect.h / 2;
    const u = dx * Math.cos(rect.a) + dy * Math.sin(rect.a);
    const v = -dx * Math.sin(rect.a) + dy * Math.cos(rect.a);
    return Math.abs(u) <= rect.w / 2 && Math.abs(v) <= rect.h / 2;
  };
  // Deliberately preserves corner-containment collision, not full SAT overlap.
  const collision = (a, b) => corners(a).some(p => contains(b, p)) || corners(b).some(p => contains(a, p));
  const parking = ctx => {
    const {root, level, state, emit, active, finish, reject, random} = ctx;
    const first = level === 15 ? 0 : 2;
    let stage = first, car, moving = [], lost = false, stopped = false, frame = 0, resetTimer = 0;
    let last = performance.now();
    const keys = new Set();
    const virtualKeys = new Set();
    const aliases = {arrowleft: "a", arrowup: "w", arrowdown: "s", arrowright: "d"};
    // Preserve each input owner; releasing one alias must not release another.
    const held = control => {
      if (virtualKeys.has(control)) return true;
      for (const key of keys) if ((aliases[key] || key) === control) return true;
      return false;
    };
    ctx.heading("Please", level === 15 ? "Park the car" : "Parallel park");
    const board = element("div", root, {width: "400px", height: "400px", position: "relative", overflow: "hidden", margin: "4px auto", background: "#64716d", touchAction: "none"});
    board.className = "neal-parking-board";
    const scene = canvas(board);
    const paint = scene.getContext("2d");
    const carNode = element("div", board, {position: "absolute", width: "80px", height: "50px", border: "3px solid #e1f4fd", boxSizing: "border-box", borderRadius: "12px", background: "#1473b5", color: "white", display: "grid", placeItems: "center", pointerEvents: "none", transformOrigin: "center"}, "CAR ▸");
    carNode.className = "neal-parking-car";
    // Source wheel geometry, drawn above the opaque authored body so steering is visible.
    const frontWheels = [-1, 1].map(side => {
      const wheel = element("span", carNode, {position: "absolute", left: "52px", top: `${25 + side * 50 / 3 - 6}px`, width: "16px", height: "6px", background: "#000", transformOrigin: "center"});
      wheel.className = "neal-parking-front-wheel";
      return wheel;
    });
    const status = label(root, "neal-parking-status", "");
    const controls = element("div", root, {display: "flex", justifyContent: "center", gap: "4px", touchAction: "none"});
    const stopKeys = () => {keys.clear(); virtualKeys.clear();};
    for (const [name, key, glyph] of [["Steer left","a","←"],["Drive forward","w","↑"],["Reverse","s","↓"],["Steer right","d","→"]]) {
      const control = action(ctx, controls, name, glyph);
      control.style.touchAction = "none";
      control.addEventListener("pointerdown", event => {
        if (!active()) return;
        event.preventDefault(); control.setPointerCapture(event.pointerId); virtualKeys.add(key);
        emit("parking_control", {control: key, pressed: true, input_source: "pointer"});
      });
      const release = () => {virtualKeys.delete(key); emit("parking_control", {control: key, pressed: false, input_source: "pointer"});};
      control.addEventListener("pointerup", release); control.addEventListener("pointercancel", release);
    }
    const keyboard = pressed => event => {
      const key = event.key.toLowerCase();
      const control = aliases[key] || key;
      if (!["w", "a", "s", "d"].includes(control) || !active()) return;
      event.preventDefault();
      if (pressed) keys.add(key); else keys.delete(key);
      if (!event.repeat) emit("parking_control", {control, pressed, physical_key: key, input_source: "keyboard"});
    };
    const down = keyboard(true), up = keyboard(false);
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    window.addEventListener("blur", stopKeys);
    const reset = () => {
      clearTimeout(resetTimer); lost = false;
      const [x, y, a] = parkingStages[stage].start;
      car = {x, y, a, w: 80, h: 50, v: 0, steer: 0, momentum: 0, throttleTime: 0};
      state.progress = stage - first;
      status.textContent = `Stage ${stage - first + 1}/2 · Hold arrows or WASD to drive`;
      board.dataset.phase = "driving"; draw();
    };
    const load = () => {
      moving = parkingStages[stage].moving.map(([x, y, w, h, a]) => ({x, y, w, h, a, spawnX: x, spawnY: y, off: false, timer: 0}));
      reset();
    };
    function draw() {
      paint.fillStyle = "#69736f"; paint.fillRect(0, 0, 400, 400);
      const [x, y, w, h] = parkingStages[stage].slot;
      paint.fillStyle = "#376945"; paint.fillRect(x, y, w, h);
      paint.strokeStyle = "#f2ff9b"; paint.lineWidth = 2; paint.strokeRect(x, y, w, h);
      paint.fillStyle = "#fff"; paint.font = "bold 22px sans-serif"; paint.fillText("P", x + w / 2 - 7, y + h / 2 + 8);
      for (const [ox, oy, ow, oh] of parkingStages[stage].obstacles) {
        paint.fillStyle = "#363b43"; paint.fillRect(ox, oy, ow, oh);
        paint.strokeStyle = "#b7bdc3"; paint.strokeRect(ox + 1, oy + 1, ow - 2, oh - 2);
      }
      for (const other of moving) {
        paint.save(); paint.translate(other.x + other.w / 2, other.y + other.h / 2); paint.rotate(other.a);
        paint.fillStyle = "#be563b"; paint.fillRect(-other.w/2, -other.h/2, other.w, other.h);
        paint.fillStyle = "#bcd7dc"; paint.fillRect(-other.w/2+8, -other.h/2+12, other.w-16, 20); paint.restore();
      }
      Object.assign(carNode.style, {left: `${car.x}px`, top: `${car.y}px`, transform: `rotate(${car.a}rad)`, background: lost ? "#9b2020" : "#1473b5"});
      for (const wheel of frontWheels) wheel.style.transform = `rotate(${car.steer}rad)`;
    }
    const step = h => {
      if (held("a")) car.steer -= .05 * h;
      else if (held("d")) car.steer += .05 * h;
      else {
        car.steer -= Math.sign(car.steer) * .05 * h;
        if (Math.abs(car.steer) < .05 * h) car.steer = 0;
      }
      car.steer = clamp(car.steer, -Math.PI/4, Math.PI/4);
      if (held("w") || held("s")) {
        car.throttleTime = Math.min(120, car.throttleTime + h);
        car.v += (held("w") ? 1 : -1) * .02 * (1 + 3 * car.throttleTime / 120) * h;
        car.momentum = Math.min(.9, car.momentum + .15 * h);
      } else {
        car.throttleTime = 0;
        car.v *= Math.pow(.88 + .1 * car.momentum, h); car.momentum *= Math.pow(.97, h);
      }
      car.v = clamp(car.v * Math.pow(.985, h), -2, 2);
      if (Math.abs(car.v) < .005) car.v = 0;
      if (Math.abs(car.v) > .1) car.a += car.steer * car.v * .03 * h;
      car.x = clamp(car.x + Math.cos(car.a) * car.v * h, 0, 320);
      car.y = clamp(car.y + Math.sin(car.a) * car.v * h, 0, 350);
      if (car.x === 0 || car.x === 320 || car.y === 0 || car.y === 350) car.v = 0;
      for (const other of moving) {
        if (other.off) {
          other.timer += h / 60;
          if (other.timer >= 5 + 2 * random()) Object.assign(other, {x: other.spawnX, y: other.spawnY, off: false, timer: 0});
        } else {
          other.x += Math.cos(other.a + 3*Math.PI/2) * 2*h;
          other.y += Math.sin(other.a + 3*Math.PI/2) * 2*h;
          if (other.x < -100 || other.x > 500 || other.y < -100 || other.y > 500) Object.assign(other, {off: true, timer: 0});
        }
      }
      const hit = parkingStages[stage].obstacles.some(([x,y,w,h]) => collision(reducedCar(car), {x,y,w,h,a:0})) || moving.some(other => collision(reducedCar(car), reducedCar(other)));
      if (hit) {
        lost = true; board.dataset.phase = "collision";
        status.textContent = "Collision · Resetting this stage in 2 seconds";
        reject("parking_collision"); resetTimer = setTimeout(() => {if (active()) reset();}, 2000);
      }
    };
    const tick = now => {
      if (stopped) return;
      const delta = now - last; last = now;
      if (active() && !lost) step(delta / 16.67);
      draw(); frame = requestAnimationFrame(tick);
    };
    ctx.verifyFooter(() => {
      const [x,y,w,h] = parkingStages[stage].slot;
      if (car.x >= x-5 && car.x+80 <= x+w+5 && car.y >= y-5 && car.y+50 <= y+h+5) {
        state.progress = stage - first + 1;
        emit("parking_stage_complete", {stage: stage-first+1});
        if (stage === first+1) finish("both_parking_stages_complete");
        else {stage++; load();}
      } else reject("car_not_in_parking_space");
    });
    const stopDynamic = () => {
      stopped = true; cancelAnimationFrame(frame); clearTimeout(resetTimer); stopKeys();
      window.removeEventListener("keydown", down); window.removeEventListener("keyup", up);
      window.removeEventListener("blur", stopKeys); window.removeEventListener("pagehide", stopDynamic);
    };
    window.addEventListener("pagehide", stopDynamic);
    load(); frame = requestAnimationFrame(tick);
    return {refresh: reset, stopDynamic};
  };
  window.nealTasks[15] = {assets: [], render: parking};
  window.nealTasks[26] = {assets: [], render: parking};

  window.nealTasks[17] = {assets: [], render(ctx) {
    const {root, active, state, emit, finish, reject} = ctx;
    ctx.heading("Draw a", "Perfect circle");
    const board = canvas(root, 1000); board.className = "neal-circle-canvas";
    board.setAttribute("aria-label", "Draw a circle around the center dot");
    const paint = board.getContext("2d");
    const output = label(root, "neal-circle-score", "0.0% · Draw around the center dot");
    let drawing = false, initialRadius = 0, previous, previousRadius = 0, previousAngle = 0;
    let direction = 0, angleSum = 0, weighted = 0, score = 0, best = 0, timer = 0;
    const display = message => {
      state.progress = score / 10;
      output.textContent = `${(score/10).toFixed(1)}% · Best ${(best*100).toFixed(1)}%${message ? ` · ${message}` : ""}`;
    };
    const clear = () => {
      clearTimeout(timer); drawing = false; score = 0; angleSum = 0; weighted = 0; direction = 0;
      paint.fillStyle = "#fbfaf5"; paint.fillRect(0,0,1000,1000);
      paint.fillStyle = "#dc584a"; paint.beginPath(); paint.arc(500,500,8,0,2*Math.PI); paint.fill();
      display("Draw around the center dot");
    };
    const complete = () => {
      drawing = false; clearTimeout(timer); best = Math.max(best, score/1000);
      display("Circle complete"); emit("circle_complete", {score: score/10});
    };
    const end = () => {
      if (!drawing) return;
      if (angleSum > 335) complete();
      else {drawing = false; score = 0; clearTimeout(timer); display("Draw a full circle"); emit("circle_incomplete");}
    };
    board.addEventListener("pointerdown", event => {
      if (!active()) return;
      event.preventDefault(); clear();
      previous = point(event, board, 1000);
      initialRadius = previousRadius = Math.hypot(previous[0]-500, previous[1]-500);
      previousAngle = Math.atan2(previous[1]-500, previous[0]-500)*180/Math.PI;
      if (initialRadius < 100) {display("Start farther from the center"); return;}
      drawing = true; board.setPointerCapture(event.pointerId); emit("circle_start");
      timer = setTimeout(() => {if (drawing) {end(); display("Too slow");}}, 7000);
    });
    board.addEventListener("pointermove", event => {
      if (!drawing || !active()) return;
      const next = point(event, board, 1000);
      if (Math.hypot(next[0]-previous[0], next[1]-previous[1]) < 6) return;
      const radius = Math.hypot(next[0]-500, next[1]-500);
      const angle = Math.atan2(next[1]-500, next[0]-500)*180/Math.PI;
      let delta = ((angle-previousAngle+540)%360)-180;
      if (!direction && Math.abs(delta) >= .1) direction = Math.sign(delta);
      if (radius < 30 || (direction && delta * direction < 0)) {
        end(); display(radius < 30 ? "Too close to the center" : "Keep one direction"); return;
      }
      const arc = Math.min(Math.abs(delta), 360-angleSum);
      weighted += arc * Math.max(0, 1-Math.abs((previousRadius+radius)/2-initialRadius)/initialRadius);
      angleSum += arc;
      score = angleSum ? Math.min(999, Math.round(1000*weighted/angleSum)) : 0;
      paint.strokeStyle = "#357caf"; paint.lineWidth = 6; paint.lineCap = "round";
      paint.beginPath(); paint.moveTo(...previous); paint.lineTo(...next); paint.stroke();
      previous = next; previousRadius = radius; previousAngle = angle; display("");
      if (angleSum >= 360) complete();
    });
    board.addEventListener("pointerup", end); board.addEventListener("pointercancel", end);
    board.addEventListener("lostpointercapture", end);
    ctx.verifyFooter(() => score >= 940 || best >= 940 ? finish("circle_accuracy_at_least_94_percent") : reject("circle_accuracy_below_94_percent"));
    clear();
    return {refresh: clear, stopDynamic: () => {drawing = false; clearTimeout(timer);}};
  }};

  window.nealTasks[25] = {assets: [], render(ctx) {
    const {root, state, active, emit, finish, reject, random} = ctx;
    ctx.heading("Please show your", "Creativity");
    const toolbar = element("div", root, {display: "flex", flexWrap: "wrap", gap: "4px", padding: "6px 0"});
    let tool = "brush", count = 0, lastColorNonblack = false, drawing = false, previous;
    const used = {brush: false, spray: false, pencil: false};
    const choices = [];
    for (const name of ["brush", "spray", "pencil", "eraser"]) {
      const control = action(ctx, toolbar, name[0].toUpperCase()+name.slice(1));
      control.addEventListener("click", () => {if (active()) {tool = name; updateTools(); emit("drawing_tool", {tool});}});
      choices.push([name, control]);
    }
    function updateTools() {for (const [name,node] of choices) {node.setAttribute("aria-pressed", String(tool === name)); node.style.background = tool === name ? "#cadcf3" : "white";}}
    const color = element("input", toolbar, {width: "48px", height: "38px"});
    color.type = "color"; color.value = "#000000"; color.setAttribute("aria-label", "Drawing color");
    const board = canvas(root); board.className = "neal-creativity-canvas";
    board.style.border = "1px solid #bbb";
    const paint = board.getContext("2d");
    const status = label(root, "neal-drawing-status", "Use your imagination");
    const clear = () => {paint.fillStyle = "white"; paint.fillRect(0,0,400,400); state.progress = count;};
    const stroke = next => {
      paint.strokeStyle = tool === "eraser" ? "white" : color.value;
      paint.fillStyle = paint.strokeStyle; paint.lineCap = "round";
      if (tool === "spray") {
        for (let i=0; i<20; i++) paint.fillRect(next[0]+(random()-.5)*40,next[1]+(random()-.5)*40,1,1);
      } else {
        paint.lineWidth = tool === "pencil" ? 3 : 10;
        const passes = tool === "pencil" ? 3 : 1;
        for (let i=0; i<passes; i++) {
          const jitter = tool === "pencil" ? (random()-.5)*2 : 0;
          paint.beginPath(); paint.moveTo(previous[0]+jitter,previous[1]+jitter); paint.lineTo(next[0]+jitter,next[1]+jitter); paint.stroke();
        }
      }
      previous = next;
    };
    board.addEventListener("pointerdown", event => {
      if (!active()) return;
      event.preventDefault(); board.setPointerCapture(event.pointerId);
      count++; used[tool] = true; lastColorNonblack = color.value !== "#000000";
      state.progress = count; drawing = true; previous = point(event,board,400);
      stroke([previous[0]+.01,previous[1]+.01]);
      status.textContent = `${count} strokes`; emit("drawing_stroke", {tool, color: color.value, strokes: count});
    });
    board.addEventListener("pointermove", event => {if (drawing && active()) stroke(point(event,board,400));});
    const up = () => {drawing = false;};
    board.addEventListener("pointerup", up); board.addEventListener("pointercancel", up); board.addEventListener("lostpointercapture", up);
    ctx.verifyFooter(() => count > 10 && Object.values(used).every(Boolean) && lastColorNonblack ? finish("creative_tools_strokes_and_color") : reject("creativity_requirements_not_met"));
    clear(); updateTools();
    return {refresh: clear, stopDynamic: up};
  }};

  // The reference price process uses ordinary JS multiplication, not Math.imul.
  const priceHash = n => {
    let value = (1 * 2654435761) >>> 0;
    value = (value ^ Math.abs(Math.floor(n))) >>> 0;
    value = (value * 2246822519) >>> 0; value = (value ^ (value >>> 13)) >>> 0;
    value = (value * 3266489917) >>> 0; value = (value ^ (value >>> 16)) >>> 0;
    return value / 2147483648 - 1;
  };
  const smoothNoise = seconds => {
    const base = Math.floor(seconds), fraction = seconds-base;
    const smooth = n => priceHash(n)/2 + priceHash(n-1)/4 + priceHash(n+1)/4;
    const weight = (1-Math.cos(Math.PI*fraction))/2;
    return smooth(base)*(1-weight)+smooth(base+1)*weight;
  };
  window.nealTasks[28] = {assets: [], render(ctx) {
    const {root, state, active, emit, finish, reject} = ctx;
    ctx.heading("Become a", "Day trader");
    const display = element("div", root, {padding: "8px", background: "#f0f5fa", fontVariantNumeric: "tabular-nums"});
    const quote = label(display,"neal-market-price", "");
    const account = label(display,"neal-market-account", "");
    const portfolio = label(display,"neal-market-portfolio", "");
    const chart = canvas(root);
    // Source 1119/2009 use a double-resolution canvas displayed at 220px high.
    chart.width = 2 * root.offsetWidth; chart.height = 440;
    Object.assign(chart.style, {height: "220px",
      backgroundImage: "linear-gradient(90deg,rgba(255,255,255,.2) 1px,transparent 1px),linear-gradient(180deg,rgba(255,255,255,.2) 1px,transparent 1px),linear-gradient(90deg,#202020,#000)",
      backgroundSize: "28px 28px,28px 28px,100% 100%"});
    const paint = chart.getContext("2d");
    const controls = element("div", root, {display: "flex", justifyContent: "center", gap: "12px"});
    const buy = action(ctx,controls,"Buy one share","Buy 1"), sell = action(ctx,controls,"Sell one share","Sell 1");
    let cash = 500, shares = 0, cost = 0, profit = 0, price = 0, history = [], timer = 0;
    const render = () => {
      quote.textContent = `Apple · $${price} per share`;
      account.textContent = `Cash $${cash.toFixed(2)} · Shares ${shares} · Realized profit $${profit.toFixed(2)}`;
      portfolio.textContent = `Portfolio value $${(price * shares).toFixed(2)}`;
      state.progress = Math.max(cash,profit);
      buy.disabled = !active() || !history.length || cash < price;
      sell.disabled = !active() || shares === 0;
      const width = chart.width, height = chart.height;
      paint.clearRect(0,0,width,height);
      if (!history.length) return;
      const values = history.map(sample => sample.val);
      const min = .97 * Math.min(...values), max = 1.03 * Math.max(...values);
      const points = history.map((sample,index) => ({
        // The source divides by zero for one sample; this local case stays visible.
        x: history.length === 1 ? width : index * width / (history.length - 1),
        y: height - (sample.val - min) / (max - min) * height,
        buy: sample.buy, sell: sample.sell,
      }));
      paint.strokeStyle = "#00d4aa"; paint.lineWidth = 5; paint.beginPath();
      points.forEach(({x,y},index) => {
        if (index) paint.lineTo(x,y); else paint.moveTo(x,y);
      });
      paint.stroke();
      if (points.length > 1) {
        const fill = paint.createLinearGradient(0,0,0,170);
        fill.addColorStop(0,"rgba(0,212,170,.6)"); fill.addColorStop(1,"rgba(0,212,170,.1)");
        paint.lineTo(width,height); paint.lineTo(0,height); paint.closePath();
        paint.fillStyle = fill; paint.fill();
      }
      const sides = [["buy","#00d4aa"],["sell","#ff6b6b"]];
      // Same-sample trades overlap in source order; no extra time points are invented.
      for (const [side,color] of sides) {
        paint.strokeStyle = color; paint.lineWidth = 3;
        for (const p of points) if (p[side] > 0) {
          paint.beginPath(); paint.moveTo(p.x,p.y); paint.lineTo(p.x,height); paint.stroke();
        }
      }
      for (const [side,color] of sides) {
        paint.fillStyle = color;
        for (const p of points) if (p[side] > 0) {
          paint.beginPath(); paint.arc(p.x,p.y,7,0,2*Math.PI); paint.fill();
        }
      }
      const latest = points.at(-1);
      paint.beginPath(); paint.arc(latest.x,latest.y,8,0,2*Math.PI);
      paint.fillStyle = "#00d4aa"; paint.fill();
      paint.strokeStyle = "white"; paint.lineWidth = 2; paint.stroke();
      paint.font = "28px Arial"; paint.textAlign = "right"; paint.textBaseline = "middle";
      const text = `$${price.toLocaleString("en-us")}`, textX = latest.x - 20;
      const textY = latest.y - 46 < 20 ? latest.y + 40 : latest.y - 30;
      const textWidth = paint.measureText(text).width;
      paint.fillStyle = "rgba(0,0,0,.8)";
      paint.fillRect(textX-textWidth-4,textY-16,textWidth+8,32);
      paint.fillStyle = "#00d4aa"; paint.fillText(text,textX,textY);
      paint.font = "18px Arial"; paint.fillStyle = "rgba(255,255,255,.7)";
      const axisStep = max - min <= 100 ? 20 : 40;
      for (let value = Math.ceil(min / axisStep) * axisStep; value <= max; value += axisStep) {
        const y = height - (value - min) / (max - min) * height;
        paint.fillText(`$${value.toLocaleString("en-us")}`,50,y);
        paint.strokeStyle = "rgba(255,255,255,.2)"; paint.lineWidth = 1;
        paint.beginPath(); paint.moveTo(60,y); paint.lineTo(width,y); paint.stroke();
      }
    };
    const tick = () => {
      if (!active()) return;
      const seconds = performance.now()/1000;
      price = Math.round(Math.abs(300*(smoothNoise(seconds)+.5*smoothNoise(2*seconds)+.25*smoothNoise(4*seconds))/1.75+300));
      history.push({val: price, buy: 0, sell: 0}); if (history.length > 30) history.shift(); render();
    };
    buy.addEventListener("click", () => {
      if (!active() || !history.length || cash < price) return;
      shares++; cash-=price; cost+=price; history.at(-1).buy++;
      emit("trade", {side: "buy", price}); render();
    });
    sell.addEventListener("click", () => {
      if (!active() || !shares) return;
      const average = cost/shares; shares--; cost-=average; profit+=price-average; cash+=price;
      history.at(-1).sell++;
      emit("trade", {side: "sell", price}); render();
    });
    ctx.verifyFooter(() => cash >= 2500 || profit >= 2500 ? finish("day_trading_cash_or_profit_target") : reject("trading_target_not_reached"));
    const refresh = () => {cash=500;shares=0;cost=0;profit=0;history=[];render();};
    const stopDynamic = () => {clearInterval(timer); window.removeEventListener("pagehide",stopDynamic);};
    tick(); timer = setInterval(tick,500); window.addEventListener("pagehide",stopDynamic);
    return {refresh, stopDynamic};
  }};

  window.nealTasks[35] = {assets: [], render(ctx) {
    const {root, state, active, emit, finish, reject, random} = ctx;
    ctx.heading("Select the cup with the", "Ball");
    const board = element("div",root,{position:"relative",height:"235px",width:"330px",margin:"24px auto",overflow:"hidden"});
    board.className = "neal-cups-board";
    const status = label(root,"neal-cups-status", "0/3 correct");
    const positions = [0,1,2], cups = [], caps = [], balls = [];
    let ball = -1, shown = -1, reveal = true, swapping = true, round = 0;
    let interval = 0, startTimer = 0, revealTimer = 0, mountTimer = 0;
    for (let id=0; id<3; id++) {
      const cup = ctx.button("neal-cup", `Cup ${id+1}`);
      Object.assign(cup.style,{position:"absolute",left:"0",top:"70px",width:"100px",height:"150px",background:"transparent",transition:"transform 250ms",cursor:"pointer"});
      const ballNode = element("span",cup,{position:"absolute",left:"38px",bottom:"5px",width:"25px",height:"25px",borderRadius:"50%",background:"#e64e39",display:"none"});
      ballNode.className = "neal-cup-ball";
      const cap = element("span",cup,{position:"absolute",left:"5px",bottom:"0",width:"90px",height:"110px",background:"linear-gradient(90deg,#dbe4eb,#fff,#bdc9d3)",border:"2px solid #586671",borderRadius:"26px 26px 10px 10px",boxSizing:"border-box",transition:"transform 250ms"});
      cap.className = "neal-cup-cap";
      cup.addEventListener("click", () => {
        if (!active() || swapping || shown !== -1) return;
        shown = id; emit("cup_guess", {cup: id+1});
        if (id === ball) {round++; state.progress=round; if (round<3) start();}
        else {reject("wrong_cup"); revealTimer=setTimeout(() => {if(active()){shown=ball;round=0;state.progress=0;draw();start();}},3000);}
        draw();
      });
      board.append(cup); cups.push(cup); caps.push(cap); balls.push(ballNode);
    }
    function draw() {
      for (let id=0; id<3; id++) {
        cups[id].style.transform = `translateX(${positions[id]*110}px)`;
        cups[id].disabled = !active() || swapping || shown !== -1;
        const lifted = (id===ball && reveal) || id===shown;
        caps[id].style.transform = lifted ? "translateY(-75px)" : "translateY(0px)";
        // The hidden location is never serialized or present as a hidden DOM answer.
        balls[id].style.display = id===ball && lifted ? "block" : "none";
      }
      board.dataset.phase = swapping ? "shuffling" : shown===-1 ? "choose" : "revealed";
      status.textContent = `${round}/3 correct${swapping ? " · Watch carefully" : ""}`;
    }
    function start() {
      let remaining = [5,7,20][round];
      startTimer = setTimeout(() => {
        if (!active()) return;
        swapping=true;reveal=false;shown=-1;draw();
        interval=setInterval(() => {
          if (!remaining) {clearInterval(interval);swapping=false;draw();return;}
          const a=Math.floor(random()*2), others=[0,1,2].filter(id=>id!==a);
          const b=others[Math.floor(random()*2)];
          [positions[a],positions[b]]=[positions[b],positions[a]]; remaining--; draw();
          emit("cups_moved");
        },[700,500,350][round]);
      },2500);
    }
    const clearTimers = () => {clearInterval(interval);clearTimeout(startTimer);clearTimeout(revealTimer);};
    const refresh = () => {clearTimers();round=0;state.progress=0;shown=-1;swapping=true;reveal=true;draw();if(ball!==-1)start();};
    ctx.verifyFooter(() => round===3 ? finish("three_correct_cup_rounds") : reject("three_cup_rounds_required"));
    mountTimer=setTimeout(() => {ball=Math.floor(random()*3);draw();start();},500);
    const stopDynamic = () => {clearTimeout(mountTimer);clearTimers();window.removeEventListener("pagehide",stopDynamic);};
    window.addEventListener("pagehide",stopDynamic);draw();
    return {refresh,stopDynamic};
  }};

  window.nealTasks[41] = {assets: [], render(ctx) {
    const {root, active, state, emit, finish} = ctx;
    ctx.heading("Please", "Grieve");
    const scene=element("div",root,{position:"relative",width:"400px",height:"400px",margin:"4px auto",background:"linear-gradient(#809aa4 40%,#587451 40%)",overflow:"hidden",touchAction:"none"});
    scene.className="neal-mourn-scene";
    const grave=action(ctx,scene,"Clean the gravestone","RIP");
    Object.assign(grave.style,{position:"absolute",left:"165px",top:"125px",width:"85px",height:"120px",borderRadius:"40px 40px 0 0",background:"#787971",fontSize:"24px"});
    let flowers=0, candleLit=false, cleaned=false, grieve=false, zoomed=false;
    let flowerX=300,flowerY=300,brushX=120,brushY=340,drag=null,lastPointer=null,lastBrush=null,distance=0;
    const timers=new Set();
    const later=(fn,ms)=>{const timer=setTimeout(()=>{timers.delete(timer);if(active())fn();},ms);timers.add(timer);};
    const bouquet=action(ctx,scene,"Place bouquet","✿✿✿"); bouquet.className="neal-mourn-bouquet";
    Object.assign(bouquet.style,{position:"absolute",width:"64px",height:"64px",background:"transparent",color:"#abc5ff",fontSize:"28px",border:0,touchAction:"none",display:"none"});
    for (const [index,[x,y]] of [[80,70],[10,40],[15,60],[70,35]].entries()) {
      const flower=action(ctx,scene,`Pick flower ${index+1}`,"✿"); flower.className="neal-mourn-flower";
      Object.assign(flower.style,{position:"absolute",left:`${x}%`,top:`${y}%`,background:"transparent",color:"#cad8ff",fontSize:"30px",border:0});
      flower.addEventListener("click",()=>{
        if (!active() || flower.hidden) return;
        flower.hidden=true;flowers++;state.progress=flowers;
        bouquet.style.display="block";bouquet.disabled=flowers<4;drawBouquet();
        emit("flower_picked",{flower:index+1});
      });
    }
    const candle=action(ctx,scene,"Light candle","▥");
    Object.assign(candle.style,{position:"absolute",left:"115px",top:"220px",width:"40px",height:"65px",background:"transparent",fontSize:"30px",border:0});
    const status=label(root,"neal-mourn-status","Gather flowers, clean the stone and light a candle");
    const overlay=element("div",scene,{position:"absolute",inset:"0",background:"#d5d0bc",display:"none",zIndex:"2",touchAction:"none"});
    const cleanStone=element("div",overlay,{position:"absolute",left:"90px",top:"55px",width:"220px",height:"220px",background:"#bbc2bd",borderRadius:"60px 60px 0 0",display:"grid",placeItems:"center",fontSize:"32px"},"In memory");
    const dirt=canvas(cleanStone,135);dirt.className="neal-grave-dirt";
    Object.assign(dirt.style,{position:"absolute",inset:"0",height:"100%"});
    const brush=action(ctx,overlay,"Cleaning brush","▰━━━━");brush.className="neal-mourn-brush";
    Object.assign(brush.style,{position:"absolute",width:"100px",height:"60px",padding:0,border:0,background:"transparent",color:"#765537",fontSize:"25px",touchAction:"none"});
    const back=action(ctx,overlay,"Back to grave","← Back");
    Object.assign(back.style,{position:"absolute",left:"5px",top:"5px"});
    const paint=dirt.getContext("2d");
    paint.fillStyle="#68624e";paint.fillRect(0,0,135,135);
    paint.fillStyle="#899270";for(let i=0;i<38;i++)paint.fillRect((i*37)%135,(i*61)%135,12,14);
    const verify=ctx.verifyFooter(()=>{if(grieve)finish("grave_tended_and_grieving");});verify.disabled=true;
    function drawBouquet(){bouquet.style.left=`${flowerX}px`;bouquet.style.top=`${flowerY}px`;}
    function drawBrush(){brush.style.left=`${brushX}px`;brush.style.top=`${brushY}px`;}
    const check=()=>{
      if(!grieve && Math.hypot(flowerX-200,flowerY-200)<100 && candleLit && cleaned){
        grieve=true;status.textContent="Remembering…";emit("grieving_started");
        later(()=>{verify.disabled=false;status.textContent="Ready to pay your respects";},3500);
      }
    };
    grave.addEventListener("click",()=>{if(active()){zoomed=true;overlay.style.display="block";}});
    back.addEventListener("click",()=>{zoomed=false;drag=null;overlay.style.display="none";later(check,1000);});
    candle.addEventListener("click",()=>{if(active()&&!candleLit){candleLit=true;candle.textContent="♨";candle.style.color="#ffc35a";emit("candle_lit");check();}});
    const startDrag=(kind,node)=>event=>{
      if(!active() || (kind==="bouquet" && (zoomed || flowers<4)) || (kind==="brush" && !zoomed))return;
      event.preventDefault();drag=kind;lastPointer=[event.clientX,event.clientY];node.setPointerCapture(event.pointerId);emit("mourn_drag_start",{item:kind});
    };
    bouquet.addEventListener("pointerdown",startDrag("bouquet",bouquet));
    brush.addEventListener("pointerdown",startDrag("brush",brush));
    const move=event=>{
      if(!active()||!drag)return;
      event.preventDefault();const dx=event.clientX-lastPointer[0],dy=event.clientY-lastPointer[1];lastPointer=[event.clientX,event.clientY];
      if(drag==="bouquet"){flowerX+=dx;flowerY+=dy;drawBouquet();}
      else{
        brushX+=dx;brushY+=dy;drawBrush();
        const x=Math.floor((brushX+30-90)/220*135),y=Math.floor((brushY+30-55)/220*135);
        if(lastBrush&&(lastBrush[0]!==0||lastBrush[1]!==0))distance+=Math.hypot(x-lastBrush[0],y-lastBrush[1]);
        lastBrush=[x,y];paint.globalCompositeOperation="destination-out";paint.fillRect(x-4,y-4,8,8);
        if(!cleaned && distance>=900){cleaned=true;grave.style.background="#bbc2bd";status.textContent="The gravestone is clean";emit("grave_cleaned");}
      }
    };
    const end=()=>{if(drag){const item=drag;drag=null;emit("mourn_drag_end",{item});if(item==="bouquet")check();}};
    window.addEventListener("pointermove",move);window.addEventListener("pointerup",end);window.addEventListener("pointercancel",end);
    const stopDynamic=()=>{drag=null;for(const timer of timers)clearTimeout(timer);timers.clear();window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",end);window.removeEventListener("pointercancel",end);window.removeEventListener("pagehide",stopDynamic);};
    window.addEventListener("pagehide",stopDynamic);drawBouquet();drawBrush();
    // The original component has no refresh handler: preserve the scene.
    return {refresh:()=>{state.progress=flowers;},stopDynamic};
  }};
})();
