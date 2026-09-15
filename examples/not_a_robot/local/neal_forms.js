/* Captured statement, eye-exam, and logo-text instances.
 * Image clues are CSS crops of unchanged private reference screenshots, not
 * rendered evaluator answers. Input normalization and refresh are local rules.
 */
"use strict";

(() => {
  const screenshotCrop = (asset, filename, sourceWidth, box, description) => {
    const [x, y, width, height] = box;
    const crop = document.createElement("div");
    crop.className = "neal-reference-crop";
    Object.assign(crop.style, {position: "relative", width: "100%",
      aspectRatio: `${width} / ${height}`, overflow: "hidden"});
    const image = document.createElement("img");
    image.src = asset(filename);
    image.alt = description;
    image.draggable = false;
    Object.assign(image.style, {position: "absolute", display: "block", maxWidth: "none",
      width: `${sourceWidth / width * 100}%`, height: "auto", left: `${-x / width * 100}%`,
      top: `${-y / height * 100}%`, userSelect: "none", pointerEvents: "none"});
    crop.append(image);
    return crop;
  };

  window.nealTasks[14] = {
    assets: [],
    render({root, button, state, emit, active, finish}) {
      // All 56 labels and their order come from attempt_301's initial AX tree.
      const labels = [
        "I'm a robot", "I'm not not a robot", "I'm potentially a robot", "I'm a chatbot",
        "I'm a robocaller", "I dream of electric sheep", "I'm not a human",
        "I'm a coffee machine with a subscription service", "I'm barely a robot", "I'm a TI-95",
        "I used to be a robot", "I'm a robot", "I'm a digital assistant", "I'm in beta",
        "I'm a military drone", "I'm a neural network", "I'm a sentient toaster",
        "I'm running on AA batteries", "I'm a turing machine", "I'm not carbon based",
        "I'm an AI Girlfriend", "I'm a smart fridge", "I'm a CAPTCHA", "I'm a roomba",
        "I'm a fax machine", "I'm a trading algorithm", "I'm stuck in a loop",
        "I'm running on dial-up", "I'm an outdated plugin", "I'm Cleverbot", "I'm Akinator",
        "I'm a microwave", "I'm a pool cleaning bot", "I'm not a robot",
        "I'm an abandoned Smart TV", "I'm a HP OfficeJet Pro 8139e Wireless All-in-One Printer",
        "I'm legally a blender", "I'm a global positioning system", "I'm a spam filter",
        "I'm a cryptocurrency miner", "I'm an elevator music generator", "I'm spyware from the 90s",
        "I'm a insecure thermostat", "I'm a barcode scanner", "I'm a voice assistant",
        "I'm a 404 error message", "01010010 01001111 01000010 01001111 01010100",
        "I'm a pop-up ad", "I'm an autocorrect algorithm", "I'm a cyborg", "I'm a vending machine",
        "I'm Stockfish", "I'm a warehouse conveyor belt", "I'm Arnold Schwarzenegger",
        "I'm an easy-bake oven", "I'm a hot single in your area",
      ];
      root.classList.add("neal-statements");
      Object.assign(root.style, {width: "600px", height: "590px", overflowY: "scroll",
        padding: "14px", border: "0"});
      const timers = new Map();
      const controls = [];
      for (const [index, label] of labels.entries()) {
        const card = document.createElement("div");
        card.className = "neal-statement-card";
        Object.assign(card.style, {display: "flex", alignItems: "center", justifyContent: "space-between",
          minHeight: "76px", padding: "0 14px", marginBottom: "10px", border: "1px solid #ddd",
          borderRadius: "2px", background: "#fafafa", boxShadow: "0 1px 4px #00000016"});
        const checkbox = button("neal-checkbox-target", label);
        checkbox.setAttribute("role", "checkbox");
        checkbox.setAttribute("aria-checked", "false");
        Object.assign(checkbox.style, {display: "flex", alignItems: "center", gap: "14px",
          textAlign: "left", background: "transparent"});
        const mark = document.createElement("span");
        mark.className = "neal-checkbox-mark";
        mark.style.flex = "0 0 28px";
        const caption = document.createElement("span");
        caption.textContent = label;
        checkbox.append(mark, caption);
        const logo = document.createElement("div");
        logo.className = "neal-recaptcha-mark";
        logo.style.flex = "0 0 57px";
        // Authored branding approximation, shared in appearance with level 01.
        logo.innerHTML = '<svg viewBox="0 0 36 36" aria-hidden="true"><path fill="#6083c5" d="M31 14A14 14 0 0 0 7 6L4 3v13h13l-5-5a8 8 0 0 1 14 3h5Z"/><path fill="#5264a3" d="M5 22a14 14 0 0 0 24 8l3 3V20H19l5 5a8 8 0 0 1-14-3H5Z"/></svg><span>reCAPTCHA</span>';
        card.append(checkbox, logo);
        root.append(card);
        controls.push({checkbox, mark});
        checkbox.addEventListener("click", () => {
          if (!active() || timers.has(index)) return;
          mark.classList.add("loading");
          emit("checkbox_loading", {index});
          // Loading is observed; 650 ms and distractor restoration are inferred.
          timers.set(index, setTimeout(() => {
            timers.delete(index);
            if (!active()) return;
            mark.classList.remove("loading");
            if (label === "I'm not a robot") {
              mark.classList.add("checked");
              checkbox.setAttribute("aria-checked", "true");
              state.progress = 1;
              finish("exact_robot_statement_checked");
            } else {
              // The source never captured final distractor feedback. Do not
              // relabel that observation as a confirmed original rejection.
              emit("local_distractor_reset", {index, original_feedback: "unknown"});
            }
          }, 650));
        });
      }
      const stopDynamic = () => {
        timers.forEach((timer) => clearTimeout(timer));
        timers.clear();
        controls.forEach(({mark}) => mark.classList.remove("loading"));
      };
      return {stopDynamic, refresh: () => {
        stopDynamic();
        controls.forEach(({checkbox, mark}) => {
          checkbox.setAttribute("aria-checked", "false");
          mark.classList.remove("loading", "checked");
        });
        root.scrollTop = 0;
      }};
    },
  };

  window.nealTasks[24] = {
    assets: ["level24_stage01.jpg", "level24_stage02.jpg", "level24_stage03.jpg", "level24_stage04.jpg"],
    render({root, asset, heading, newGrid, newTile, verifyFooter, state, emit, active, finish, reject}) {
      let stage = 0;
      let input = null;
      const selected = new Set();
      const prompts = ["Please type the letters in the last row", "Please type the number you see",
        "Please type the number of dots in the image", "Which square is a different color?"];
      const answers = ["EDFCZP", "8", "34"];
      // All four images and accepted inputs are from the same attempt_303.
      const boxes = [[18, 230, 447, 371], [18, 230, 447, 456], [21, 231, 442, 328]];
      const draw = () => {
        root.replaceChildren();
        root.dataset.stage = String(stage + 1);
        heading("Please complete the", "Eye Exam");
        input = null;
        selected.clear();
        if (stage < 3) {
          root.append(screenshotCrop(asset, `level24_stage0${stage + 1}.jpg`, 1280,
            boxes[stage], "Eye exam visual challenge"));
          const panel = document.createElement("div");
          Object.assign(panel.style, {padding: "12px 8px", textAlign: "center", background: "#f8f8f8"});
          const label = document.createElement("label");
          label.htmlFor = "neal-eye-answer";
          label.textContent = prompts[stage];
          Object.assign(label.style, {display: "block", marginBottom: "10px"});
          input = document.createElement("input");
          input.id = "neal-eye-answer";
          input.autocomplete = "off";
          input.spellcheck = false;
          input.setAttribute("autocapitalize", "off");
          Object.assign(input.style, {width: "180px", height: "34px", padding: "4px 8px",
            border: "1px solid #e5e5e5", borderRadius: "0", font: "16px Arial, sans-serif", color: "#111"});
          input.addEventListener("input", () => {
            if (active()) emit("text_edited", {stage: stage + 1, input_length: input.value.length});
          });
          panel.append(label, input);
          root.append(panel);
        } else {
          const prompt = document.createElement("div");
          prompt.textContent = prompts[stage];
          Object.assign(prompt.style, {padding: "22px 8px 18px", textAlign: "center"});
          root.append(prompt);
          const grid = newGrid(4);
          grid.classList.add("neal-eye-colors");
          const columns = [22, 134, 245, 357];
          const rows = [284, 396, 508, 620];
          for (let index = 0; index < 16; index++) {
            const {tile, face} = newTile(index);
            const x = columns[index % 4];
            const y = rows[Math.floor(index / 4)];
            const crop = screenshotCrop(asset, "level24_stage04.jpg", 1280,
              [x, y, 106, 106], "Color square");
            crop.style.height = "100%";
            face.append(crop);
            tile.addEventListener("click", () => {
              if (!active()) return;
              if (selected.has(index)) selected.delete(index);
              else selected.add(index);
              tile.setAttribute("aria-pressed", String(selected.has(index)));
              emit("selection_changed", {stage: 4, index, selected: selected.has(index), selected_count: selected.size});
            });
            grid.append(tile);
          }
        }
        verifyFooter(() => {
          const accepted = stage < 3 ? input.value === answers[stage] : selected.size === 1 && selected.has(12);
          emit("eye_exam_submitted", {stage: stage + 1,
            ...(input ? {input_length: input.value.length} : {selected_count: selected.size})});
          if (!accepted) {
            reject("eye_exam_input_does_not_match_reference_stage");
            return;
          }
          state.progress = stage + 1;
          emit("stage_completed", {stage: stage + 1});
          if (stage === 3) finish("four_reference_eye_exam_stages_completed");
          else {
            stage++;
            draw();
          }
        });
      };
      draw();
      return {refresh: () => {stage = 0; draw();}};
    },
  };

  window.nealTasks[33] = {
    assets: ["level33_reference.jpg"],
    render({root, asset, button, state, emit, active, finish, reject}) {
      root.classList.add("neal-text-card");
      const form = document.createElement("form");
      const label = document.createElement("label");
      label.htmlFor = "neal-answer";
      label.textContent = "Enter the text below";
      const picture = screenshotCrop(asset, "level33_reference.jpg", 1265,
        [31, 177, 422, 134], "Noisy brand-letter challenge");
      picture.classList.add("neal-brand-picture");
      const row = document.createElement("div");
      row.className = "neal-input-row";
      const input = document.createElement("input");
      input.id = "neal-answer";
      input.placeholder = "Answer";
      input.autocomplete = "off";
      input.spellcheck = false;
      input.setAttribute("autocapitalize", "off");
      const submit = button("neal-submit", "Submit");
      submit.type = "submit";
      submit.textContent = "Submit";
      row.append(input, submit);
      form.append(label, picture, row);
      root.append(form);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!active()) return;
        emit("submit", {input_length: input.value.length});
        if (input.value === "XWBKN") {
          state.progress = 1;
          finish("reference_brand_text_accepted");
        } else reject("text_does_not_match_reference_instance");
      });
      input.addEventListener("input", () => {
        if (active()) emit("text_edited", {input_length: input.value.length});
      });
      return {refresh: () => {input.value = "";}};
    },
  };
})();
