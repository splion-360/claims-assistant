// Phone-side camera for a sweep whose conversation and inventory run on another machine.
const codeInput = document.querySelector("#code");
const pairButton = document.querySelector("#pair");
const cameraButton = document.querySelector("#camera");
const pairingPanel = document.querySelector("#pairing");
const preview = document.querySelector("#preview");
const statusEl = document.querySelector("#status");
const canvas = document.createElement("canvas");

let socket = null;
let cameraConfig = null;
let stream = null;
let frameTimer = null;
let screenLock = null;

function setStatus(text, bad) {
  statusEl.textContent = text;
  statusEl.classList.toggle("bad", Boolean(bad));
}

// A sweep runs for minutes without a touch, so the phone would otherwise lock mid-capture.
async function holdScreenAwake() {
  if (screenLock || !navigator.wakeLock?.request) return;
  try {
    screenLock = await navigator.wakeLock.request("screen");
    screenLock.addEventListener("release", () => { screenLock = null; });
  } catch { screenLock = null; }
}

function releaseScreen() {
  const held = screenLock;
  screenLock = null;
  held?.release().catch(() => {});
}

function pair() {
  const code = codeInput.value.trim();
  if (!/^[0-9]{6}$/.test(code)) { setStatus("Enter the six digits shown on the computer.", true); return; }
  pairButton.disabled = true;
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/ws/capture?code=${encodeURIComponent(code)}`);
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "paired") {
      cameraConfig = message.camera;
      pairingPanel.hidden = true;
      cameraButton.hidden = false;
      setStatus("Paired. Start the sweep when you are ready.");
    } else if (message.type === "error") {
      setStatus(message.message, true);
    }
  };
  socket.onclose = () => {
    stopCamera();
    pairButton.disabled = false;
    pairingPanel.hidden = false;
    cameraButton.hidden = true;
    setStatus("Disconnected. Check the code is still shown on the computer, then pair again.", true);
  };
  socket.onerror = () => setStatus("Could not reach the sweep. Is the computer still running it?", true);
}

function sendFrame() {
  if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 512000) return;
  if (!preview.videoWidth) return;
  const width = cameraConfig.frame_width;
  canvas.width = width;
  canvas.height = Math.round((preview.videoHeight / preview.videoWidth) * width);
  canvas.getContext("2d").drawImage(preview, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", cameraConfig.jpeg_quality);
  socket.send(JSON.stringify({ type: "video", data: dataUrl.split(",")[1] }));
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { setStatus("This browser cannot open a camera.", true); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: cameraConfig.ideal_width }, height: { ideal: cameraConfig.ideal_height }, facingMode: "environment" },
    });
    preview.srcObject = stream;
    preview.hidden = false;
    stream.getVideoTracks().forEach((track) => track.addEventListener("ended", stopCamera, { once: true }));
    socket.send(JSON.stringify({ type: "camera_state", enabled: true }));
    frameTimer = window.setInterval(sendFrame, cameraConfig.frame_interval_ms);
    cameraButton.textContent = "Stop sweep";
    cameraButton.classList.add("stop");
    await holdScreenAwake();
    setStatus("Sweeping. Watch the inventory fill in on the computer.");
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    setStatus(denied ? "Camera access was denied. Allow it for this page." : `Camera failed: ${error.message}`, true);
    stopCamera();
  }
}

function stopCamera() {
  releaseScreen();
  if (frameTimer) window.clearInterval(frameTimer);
  frameTimer = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  preview.srcObject = null;
  preview.hidden = true;
  cameraButton.textContent = "Start sweep";
  cameraButton.classList.remove("stop");
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "camera_state", enabled: false }));
    setStatus("Sweep stopped.");
  }
}

pairButton.addEventListener("click", pair);
codeInput.addEventListener("keydown", (event) => { if (event.key === "Enter") pair(); });
cameraButton.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && stream) holdScreenAwake();
});
window.addEventListener("pagehide", stopCamera);
