// WebXR AR depth capture, shared by the sweep page and the phone capture page.
// Depth gives real metric scale; a secure context is all it needs, so it works
// over a tunnel as well as over a forwarded localhost port.
async function arDepthSupported() {
  try { return Boolean(navigator.xr && window.XRWebGLBinding && await navigator.xr.isSessionSupported("immersive-ar")); }
  catch { return false; }
}

function depthGrid(depthInfo, width, height) {
  // Sample depth on a coarse grid of normalized view points, in millimetres (0 = unknown).
  const mm = new Uint16Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      let metres = 0;
      try { metres = depthInfo.getDepthInMeters((col + 0.5) / width, (row + 0.5) / height); } catch { metres = 0; }
      mm[row * width + col] = Number.isFinite(metres) ? Math.min(65535, Math.max(0, Math.round(metres * 1000))) : 0;
    }
  }
  let binary = "";
  const bytes = new Uint8Array(mm.buffer);
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}

// Build one frame's JPEG and its depth grid from the AR camera texture.
function arFrameMessage(gl, framebuffer, canvas, texture, view, depthInfo, config) {
  const { width, height } = view.camera;
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const pixels = new Uint8ClampedArray(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  // readPixels returns rows bottom-up; flip into a top-down canvas before scaling to the JPEG size.
  const source = document.createElement("canvas");
  source.width = width; source.height = height;
  const flipped = new ImageData(width, height);
  for (let row = 0; row < height; row++) {
    flipped.data.set(pixels.subarray((height - 1 - row) * width * 4, (height - row) * width * 4), row * width * 4);
  }
  source.getContext("2d").putImageData(flipped, 0, 0);
  const jpegWidth = config.frame_width;
  const jpegHeight = Math.round(height * jpegWidth / width);
  canvas.width = jpegWidth; canvas.height = jpegHeight;
  canvas.getContext("2d").drawImage(source, 0, 0, jpegWidth, jpegHeight);
  // Intrinsics in JPEG pixels from the column-major projection matrix.
  const p = view.projectionMatrix;
  return {
    type: "video",
    data: canvas.toDataURL("image/jpeg", config.jpeg_quality).split(",")[1],
    depth: {
      width: config.depth_grid_width, height: config.depth_grid_height,
      mm: depthGrid(depthInfo, config.depth_grid_width, config.depth_grid_height),
      fx: p[0] * jpegWidth / 2, fy: p[5] * jpegHeight / 2,
      cx: (1 - p[8]) * jpegWidth / 2, cy: (1 + p[9]) * jpegHeight / 2,
      // Camera-to-world transform in the gravity-aligned local space; room geometry needs it.
      pose: Array.from(view.transform.matrix),
    },
  };
}

// Run an AR session, calling send() with each frame's message at the configured cadence.
async function startArSession(config, send, onEnd) {
  const session = await navigator.xr.requestSession("immersive-ar", {
    requiredFeatures: ["depth-sensing", "camera-access", "dom-overlay"],
    depthSensing: { usagePreference: ["cpu-optimized"], dataFormatPreference: ["luminance-alpha", "float32"] },
    domOverlay: { root: document.body },
  });
  const glCanvas = document.createElement("canvas");
  const gl = glCanvas.getContext("webgl2", { xrCompatible: true });
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const space = await session.requestReferenceSpace("local");
  const binding = new XRWebGLBinding(session, gl);
  const framebuffer = gl.createFramebuffer();
  const canvas = document.createElement("canvas");
  let lastSent = -Infinity;
  const onFrame = (time, frame) => {
    session.requestAnimationFrame(onFrame);
    if (time - lastSent < config.frame_interval_ms) return;
    const view = frame.getViewerPose(space)?.views?.[0];
    if (!view?.camera) return;
    const depthInfo = frame.getDepthInformation(view);
    const texture = binding.getCameraImage(view.camera);
    if (!depthInfo || !texture) return;
    lastSent = time;
    send(arFrameMessage(gl, framebuffer, canvas, texture, view, depthInfo, config));
  };
  session.requestAnimationFrame(onFrame);
  session.addEventListener("end", onEnd, { once: true });
  return session;
}
