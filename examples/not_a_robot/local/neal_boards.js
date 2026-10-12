/* Interactive reconstructions of the captured crafting, sliding, and math boards.
 * Recipe material conservation and board state determine success, never the
 * number of actions or agreement with one recorded solution route.
 */
"use strict";

window.nealTasks[21] = {
  assets: ["level21_arrow.png", "level21_log.webp", "level21_planks.webp",
    "level21_button.webp", "level21_diamond.webp", "level21_stick.webp", "level21_pickaxe.webp"],
  render({root, asset, button, heading, verifyFooter, state, emit, active, finish, reject}) {
    heading("Select items to craft a", "Diamond Pickaxe");
    root.style.position = "relative";
    const panel = document.createElement("div");
    panel.className = "neal-crafting-panel";
    Object.assign(panel.style, {background: "#c6c6c6", padding: "28px 20px 46px", marginTop: "4px",
      fontFamily: "monospace", userSelect: "none"});
    const title = document.createElement("div");
    title.textContent = "Crafting";
    Object.assign(title.style, {fontSize: "28px", letterSpacing: "1px", marginBottom: "8px"});
    const workbench = document.createElement("div");
    Object.assign(workbench.style, {display: "grid", gridTemplateColumns: "3fr 2fr 1fr", alignItems: "center"});
    const crafting = document.createElement("div");
    crafting.className = "neal-crafting-grid";
    Object.assign(crafting.style, {display: "grid", gridTemplateColumns: "repeat(3, 1fr)"});
    const arrow = document.createElement("img");
    arrow.src = asset("level21_arrow.png");
    arrow.alt = "Crafting output";
    Object.assign(arrow.style, {width: "55%", margin: "auto", imageRendering: "pixelated"});
    const inventoryTitle = document.createElement("div");
    inventoryTitle.textContent = "Inventory";
    Object.assign(inventoryTitle.style, {fontSize: "28px", margin: "38px 0 8px", letterSpacing: "1px"});
    const inventoryNode = document.createElement("div");
    inventoryNode.className = "neal-crafting-inventory";
    Object.assign(inventoryNode.style, {display: "grid", gridTemplateColumns: "repeat(6, 1fr)"});
    const items = {
      log: ["Oak Log", "level21_log.webp"],
      planks: ["Oak Planks", "level21_planks.webp"],
      button: ["Oak Button", "level21_button.webp"],
      diamond: ["Diamond", "level21_diamond.webp"],
      stick: ["Stick", "level21_stick.webp"],
      pickaxe: ["Diamond Pickaxe", "level21_pickaxe.webp"],
    };
    let grid = Array(9).fill(null);
    let inventory = [];
    let held = null;
    const gridButtons = [];
    const inventoryButtons = [];
    const output = button("neal-crafting-output", "Crafting output: empty");
    const ghost = document.createElement("div");
    ghost.className = "neal-crafting-held";
    Object.assign(ghost.style, {position: "absolute", width: "44px", height: "44px", zIndex: "10",
      pointerEvents: "none", display: "none"});

    const slotStyle = {position: "relative", aspectRatio: "1", background: "#8b8b8b",
      border: "3px solid", borderColor: "#373737 #fff #fff #373737", boxSizing: "border-box"};
    Object.assign(output.style, slotStyle);
    const drawItem = (node, stack, label) => {
      node.replaceChildren();
      node.dataset.item = stack?.item || "";
      node.dataset.count = String(stack?.count || 0);
      node.setAttribute("aria-label", `${label}: ${stack ? `${items[stack.item][0]} ${stack.count}` : "empty"}`);
      node.title = stack ? items[stack.item][0] : "";
      if (!stack) return;
      const image = document.createElement("img");
      image.src = asset(items[stack.item][1]);
      image.alt = "";
      image.draggable = false;
      Object.assign(image.style, {width: "80%", height: "80%", objectFit: "contain",
        position: "absolute", top: "10%", left: "10%", imageRendering: "pixelated", pointerEvents: "none"});
      node.append(image);
      if (stack.count > 1) {
        const count = document.createElement("span");
        count.textContent = String(stack.count);
        Object.assign(count.style, {position: "absolute", bottom: "-2px", right: "1px", color: "white",
          font: "bold 20px monospace", textShadow: "2px 2px #333", pointerEvents: "none"});
        node.append(count);
      }
    };
    const recipe = () => {
      const occupied = grid.map((stack, index) => stack ? index : -1).filter((index) => index >= 0);
      if (occupied.length === 1) {
        const index = occupied[0];
        const stack = grid[index];
        // Both full-stack log output and the single-plank button preview are visible in the capture.
        if (stack.item === "log") return {item: "planks", count: stack.count * 4, cost: stack.count, slots: occupied};
        if (stack.item === "planks") return {item: "button", count: stack.count, cost: stack.count, slots: occupied};
      }
      if (occupied.length === 2 && occupied[1] - occupied[0] === 3 &&
          occupied.every((index) => grid[index].item === "planks")) {
        const cost = Math.min(...occupied.map((index) => grid[index].count));
        return {item: "stick", count: cost * 4, cost, slots: occupied};
      }
      if (occupied.length === 5 && [0, 1, 2].every((index) => grid[index]?.item === "diamond") &&
          [4, 7].every((index) => grid[index]?.item === "stick")) {
        return {item: "pickaxe", count: 1, cost: 1, slots: occupied};
      }
      return null;
    };
    const draw = () => {
      gridButtons.forEach((node, index) => drawItem(node, grid[index], `Crafting cell ${index + 1}`));
      inventoryButtons.forEach((node, index) => drawItem(node, inventory[index], `Inventory slot ${index + 1}`));
      drawItem(output, recipe(), "Crafting output");
      drawItem(ghost, held, "Held stack");
      ghost.style.display = held ? "block" : "none";
      state.progress = inventory.some((stack) => stack?.item === "pickaxe") ? 1 : 0;
    };
    const locateHeld = (event) => {
      if (!active()) return;
      const rect = root.getBoundingClientRect();
      ghost.style.left = `${event.clientX - rect.left - 22}px`;
      ghost.style.top = `${event.clientY - rect.top - 22}px`;
    };
    root.addEventListener("pointermove", locateHeld);
    const transfer = (area, index, right, event) => {
      if (!active()) return;
      locateHeld(event);
      const slots = area === "crafting" ? grid : inventory;
      const stack = slots[index];
      if (right) {
        // The observed right-click places one held item, not the entire stack.
        if (!held || (stack && stack.item !== held.item)) return;
        if (stack) stack.count++;
        else slots[index] = {item: held.item, count: 1};
        if (--held.count === 0) held = null;
      } else if (!held) {
        held = stack;
        slots[index] = null;
      } else if (!stack) {
        slots[index] = held;
        held = null;
      } else if (stack.item === held.item) {
        stack.count += held.count;
        held = null;
      } else {
        // Ordinary stack swapping is a local interaction inference, not a new recipe.
        slots[index] = held;
        held = stack;
      }
      emit("inventory_changed", {area, index, button: right ? "right" : "left"});
      draw();
    };
    for (const [area, count, parent, nodes] of [
      ["crafting", 9, crafting, gridButtons], ["inventory", 6, inventoryNode, inventoryButtons],
    ]) {
      for (let index = 0; index < count; index++) {
        const node = button("neal-crafting-slot", `${area} slot ${index + 1}`);
        Object.assign(node.style, slotStyle);
        node.addEventListener("click", (event) => transfer(area, index, false, event));
        node.addEventListener("contextmenu", (event) => {
          event.preventDefault();
          transfer(area, index, true, event);
        });
        nodes.push(node);
        parent.append(node);
      }
    }
    output.addEventListener("click", (event) => {
      if (!active()) return;
      locateHeld(event);
      const result = recipe();
      if (!result || (held && held.item !== result.item)) return;
      for (const index of result.slots) {
        grid[index].count -= result.cost;
        if (grid[index].count === 0) grid[index] = null;
      }
      held = {item: result.item, count: result.count + (held?.count || 0)};
      emit("item_crafted", {item: result.item, count: result.count});
      draw();
    });
    workbench.append(crafting, arrow, output);
    panel.append(title, workbench, inventoryTitle, inventoryNode);
    root.append(panel, ghost);
    const refresh = () => {
      grid = Array(9).fill(null);
      inventory = [{item: "log", count: 2}, {item: "diamond", count: 3}, ...Array(4).fill(null)];
      held = null;
      draw();
    };
    refresh();
    verifyFooter(() => {
      // The capture stores its pickaxe in slot 2. Any inventory position is an explicit local generalization.
      if (inventory.some((stack) => stack?.item === "pickaxe")) finish("crafted_pickaxe_in_inventory");
      else reject("diamond_pickaxe_not_in_inventory");
    });
    return {refresh};
  },
};

window.nealTasks[30] = {
  assets: ["level30_background.webp"],
  render({asset, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject}) {
    heading("Reassemble the", "Stop Sign");
    const initial = [7, 4, 8, 2, 0, 6, 5, 1, 3];
    let cells = [...initial];
    const grid = newGrid(3);
    grid.classList.add("neal-sliding-grid");
    const tiles = [];
    const draw = () => {
      tiles.forEach(({tile, face}, index) => {
        const value = cells[index];
        tile.dataset.fragment = String(value);
        tile.setAttribute("aria-label", value ? `Puzzle cell ${index + 1}` : `Empty cell ${index + 1}`);
        face.style.backgroundImage = value ? `url("${asset("level30_background.webp")}")` : "none";
        face.style.backgroundSize = "300% 300%";
        if (value) face.style.backgroundPosition = `${(value - 1) % 3 * 50}% ${Math.floor((value - 1) / 3) * 50}%`;
      });
      state.progress = cells.filter((value, index) => value !== 0 && value === index + 1).length;
    };
    for (let index = 0; index < 9; index++) {
      const entry = newTile(index);
      entry.tile.addEventListener("click", () => {
        if (!active()) return;
        const empty = cells.indexOf(0);
        const distance = Math.abs(index % 3 - empty % 3) + Math.abs(Math.floor(index / 3) - Math.floor(empty / 3));
        if (distance !== 1) {
          emit("slide_ignored", {index, reason: "not_adjacent_to_empty_cell"});
          return;
        }
        [cells[index], cells[empty]] = [cells[empty], cells[index]];
        emit("tile_slid", {from: index, to: empty});
        draw();
      });
      tiles.push(entry);
      grid.append(entry.tile);
    }
    const refresh = () => {
      cells = [...initial];
      draw();
    };
    refresh();
    verifyFooter(() => {
      if (cells.every((value, index) => value === (index + 1) % 9)) finish("sliding_image_reassembled");
      else reject("sliding_image_not_reassembled");
    });
    return {refresh};
  },
};

window.nealTasks[34] = {
  assets: [],
  render({heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject}) {
    heading("Select all the squares in order of", "least to greatest");
    // Visible expressions transcribed from L34A501E6 and initial.jpg. Values, not a recorded action route, determine order.
    const expressions = [
      ['<mfrac><mrow><mn>6</mn><mi mathvariant="normal">π</mi></mrow><mn>3</mn></mfrac>', 6 * Math.PI / 3],
      ["<msup><mi>e</mi><mn>6</mn></msup>", Math.exp(6)],
      ["<mrow><msubsup><mo>∫</mo><mn>2</mn><mn>7</mn></msubsup><mi>x</mi><mi>d</mi><mi>x</mi></mrow>", (7 ** 2 - 2 ** 2) / 2],
      ["<mfrac><mn>5</mn><mn>8</mn></mfrac>", 5 / 8],
      ["<msqrt><mn>8</mn></msqrt>", Math.sqrt(8)],
      ["<mrow><mn>2</mn><mo>!</mo></mrow>", 2 * 1],
      ["<mrow><msub><mi>log</mi><mn>3</mn></msub><mo>(</mo><mn>22</mn><mo>)</mo></mrow>", Math.log(22) / Math.log(3)],
      ["<mo>∞</mo>", Infinity],
      ["<mrow><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>2</mn></mrow><mn>3</mn></munderover><mi>i</mi></mrow>", 2 + 3],
    ];
    const grid = newGrid(3);
    grid.classList.add("neal-math-grid");
    const order = [];
    const tiles = [];
    const draw = () => {
      tiles.forEach(({tile, rank}, index) => {
        const position = order.indexOf(index);
        tile.setAttribute("aria-pressed", String(position >= 0));
        rank.textContent = position >= 0 ? String(position + 1) : "";
        rank.style.display = position >= 0 ? "grid" : "none";
      });
      state.progress = order.length;
    };
    expressions.forEach(([markup], index) => {
      const {tile, face} = newTile(index);
      Object.assign(face.style, {background: "#f4f4f4", position: "relative", fontSize: "30px"});
      face.innerHTML = `<math xmlns="http://www.w3.org/1998/Math/MathML">${markup}</math>`;
      // Headless hosts may lack the mathematical-italic Greek Unicode glyphs.
      face.querySelector("math").style.fontFamily = '"DejaVu Serif", serif';
      const rank = document.createElement("span");
      rank.className = "neal-expression-rank";
      Object.assign(rank.style, {position: "absolute", right: "0", bottom: "0", width: "24px",
        height: "24px", font: "bold 16px Arial", background: "#638dd0", color: "white", placeItems: "center"});
      face.append(rank);
      tile.addEventListener("click", () => {
        if (!active()) return;
        const position = order.indexOf(index);
        if (position >= 0) order.splice(position, 1);
        else order.push(index);
        emit("expression_order_changed", {index, selected: position < 0, selected_count: order.length});
        draw();
      });
      tiles.push({tile, rank});
      grid.append(tile);
    });
    const refresh = () => {
      order.length = 0;
      draw();
    };
    refresh();
    verifyFooter(() => {
      const increasing = order.every((index, position) => position === 0 ||
        expressions[order[position - 1]][1] < expressions[index][1]);
      if (order.length === expressions.length && increasing) finish("expressions_in_ascending_order");
      else reject("expressions_not_in_ascending_order");
    });
    return {refresh};
  },
};
