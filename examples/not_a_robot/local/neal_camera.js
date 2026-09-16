/* The captured unavailable-camera branch, not facial-expression recognition.
 * The reference completed on Verify while the meter remained at zero and the
 * camera prompt stayed visible. No device, face, or media stream is requested.
 */
"use strict";

window.nealTasks[39] = {
  assets: [],
  render({root, heading, verifyFooter, state, emit, finish}) {
    heading("Please showcase the emotion", "happiness");
    const meter = document.createElement("div");
    meter.className = "neal-happiness-meter";
    meter.setAttribute("role", "progressbar");
    meter.setAttribute("aria-label", "HAPPY");
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", "0");
    meter.textContent = "0% HAPPY";
    Object.assign(meter.style, {height: "28px", marginTop: "4px", padding: "6px 4px",
      textAlign: "right", font: "bold 12px Arial, sans-serif", background: "#f0f0f0",
      borderLeft: "2px solid #ecd4d4"});
    const panel = document.createElement("div");
    panel.className = "neal-camera-unavailable";
    panel.setAttribute("aria-label", "Unable to play media.");
    Object.assign(panel.style, {height: "414px", display: "flex", alignItems: "center",
      justifyContent: "center", background: "white"});
    const prompt = document.createElement("div");
    prompt.append(document.createTextNode("Please enable your camera to "),
      document.createElement("br"), document.createTextNode("complete facial exam"));
    Object.assign(prompt.style, {width: "280px", textAlign: "center", fontWeight: "600"});
    panel.append(prompt);
    root.append(meter, panel);
    verifyFooter(() => {
      state.progress = 1;
      emit("camera_unavailable_path_verified", {camera_flow_tested: false,
        facial_expression_detection_verified: false, displayed_happiness_percent: 0});
      finish("reference_camera_unavailable_path_completed");
    });
    return {refresh: () => {
      emit("camera_unavailable_path_reset", {displayed_happiness_percent: 0});
    }};
  },
};
