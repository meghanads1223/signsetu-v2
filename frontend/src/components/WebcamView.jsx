import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const RED = "#ff2d2d";

async function createLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/models/hand_landmarker.task",
      delegate,
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

// Draws white skeleton lines, red dots, red box and label for each hand
function drawHands(ctx, hands, w, h) {
  ctx.clearRect(0, 0, w, h);
  const s = w / 960; // scale line sizes with the video size

  for (const landmarks of hands) {
    // mirror x so it matches the mirrored selfie view
    const pts = landmarks.map((p) => ({ x: (1 - p.x) * w, y: p.y * h }));

    // skeleton lines
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3 * s;
    for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[start].x, pts[start].y);
      ctx.lineTo(pts[end].x, pts[end].y);
      ctx.stroke();
    }

    // red dots on every joint
    ctx.fillStyle = RED;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6 * s, 0, Math.PI * 2);
      ctx.fill();
    }

    // red bounding box
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const pad = 24 * s;
    const minX = Math.min(...xs) - pad;
    const maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad;
    const maxY = Math.max(...ys) + pad;

    ctx.strokeStyle = RED;
    ctx.lineWidth = 3 * s;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);

    // label above the box (prediction + confidence will go here later)
    ctx.fillStyle = RED;
    ctx.font = `bold ${22 * s}px "Segoe UI", sans-serif`;
    ctx.fillText("Hand detected", minX, Math.max(minY - 10 * s, 28 * s));
  }
}

export default function WebcamView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(-1);

  const [running, setRunning] = useState(false);
  const [trackerReady, setTrackerReady] = useState(false);
  const [handsCount, setHandsCount] = useState(0);
  const [error, setError] = useState("");

  // Load the MediaPipe hand tracker once
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        let landmarker;
        try {
          landmarker = await createLandmarker(vision, "GPU");
        } catch {
          landmarker = await createLandmarker(vision, "CPU");
        }
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setTrackerReady(true);
      } catch (err) {
        console.error(err);
        if (!cancelled) setError("Could not load the hand tracker. Please check the setup files.");
      }
    }
    load();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // Runs on every video frame: detect hands, then draw
  const detectLoop = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (video && canvas && landmarker && video.readyState >= 2) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      if (video.currentTime !== lastTimeRef.current) {
        lastTimeRef.current = video.currentTime;
        const result = landmarker.detectForVideo(video, performance.now());
        drawHands(canvas.getContext("2d"), result.landmarks, canvas.width, canvas.height);
        setHandsCount(result.landmarks.length);
      }
    }
    rafRef.current = requestAnimationFrame(detectLoop);
  };

  const stopCamera = () => {
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    setHandsCount(0);
    setRunning(false);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setRunning(true);
      setError("");
      lastTimeRef.current = -1;
      rafRef.current = requestAnimationFrame(detectLoop);
    } catch (err) {
      setError("Could not access the camera. Please allow camera permission and try again.");
    }
  };

  return (
    <div className="webcam-card">
      <div className="video-frame">
        <video ref={videoRef} className="video" playsInline muted />
        <canvas ref={canvasRef} className="overlay" />
        {!running && <div className="video-placeholder">Camera is off</div>}
      </div>

      <div className="webcam-actions">
        {!running ? (
          <button className="btn btn-primary" onClick={startCamera} disabled={!trackerReady}>
            {trackerReady ? "Start camera" : "Loading hand tracker..."}
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopCamera}>
            Stop camera
          </button>
        )}
      </div>

      {running && <p className="hand-status">Hands detected: {handsCount}</p>}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}