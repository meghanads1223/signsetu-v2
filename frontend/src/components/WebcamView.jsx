import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const API_URL = "http://127.0.0.1:8000";
const RED = "#ff2d2d";

const MAX_SEGMENT_FRAMES = 90;  // about 3 seconds of one sign
const PREDICT_EVERY_MS = 400;   // live guess rate
const MIN_HAND_FRAMES = 12;     // hand frames needed before the first guess
const END_AFTER_MISSING = 12;   // frames without a hand that end a sign (about 0.4 s)
const MIN_CONFIDENCE = 0.4;     // below this the red label only says "Hand detected"
const HOLD_FINAL_MS = 3000;     // how long the final answer stays on screen

const hasHand = (frame) => frame[0][0][0] !== 0 || frame[1][0][0] !== 0;

function resultToFrame(result) {
  const zeros = () => Array.from({ length: 21 }, () => [0, 0, 0]);
  const frame = [zeros(), zeros()];
  const handedness = result.handedness ?? result.handednesses ?? [];
  const filled = [false, false];
  result.landmarks.forEach((lms, i) => {
    const name = handedness[i]?.[0]?.categoryName;
    let slot = name === "Left" ? 0 : 1;
    if (filled[slot]) {
      slot = 1 - slot;
      if (filled[slot]) return;
    }
    frame[slot] = lms.map((p) => [p.x, p.y, p.z]);
    filled[slot] = true;
  });
  return frame;
}

async function requestPrediction(frames) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  if (!res.ok) throw new Error("Prediction request failed");
  return res.json();
}

async function createLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

// White skeleton lines, red dots, red box, and the Prediction / Confidence label
function drawHands(ctx, hands, w, h, pred) {
  ctx.clearRect(0, 0, w, h);
  const s = w / 960;
  const showPred = pred && performance.now() - pred.at < pred.ttl;

  hands.forEach((landmarks, handIndex) => {
    const pts = landmarks.map((p) => ({ x: (1 - p.x) * w, y: p.y * h })); // mirrored view

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 3 * s;
    for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[start].x, pts[start].y);
      ctx.lineTo(pts[end].x, pts[end].y);
      ctx.stroke();
    }

    ctx.fillStyle = RED;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6 * s, 0, Math.PI * 2);
      ctx.fill();
    }

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

    if (handIndex === 0) {
      const lines = showPred
        ? [`Confidence: ${(pred.confidence * 100).toFixed(1)}%`, `Prediction: ${pred.label}`]
        : ["Hand detected"];
      const lineH = 26 * s;
      const top = Math.max(minY - 10 * s, lines.length * lineH);
      ctx.fillStyle = RED;
      ctx.font = `bold ${22 * s}px "Segoe UI", sans-serif`;
      lines.forEach((text, i) => {
        ctx.fillText(text, minX, top - (lines.length - 1 - i) * lineH);
      });
    }
  });
}

export default function WebcamView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const timerRef = useRef(null);
  const lastTimeRef = useRef(-1);
  const segRef = useRef({ active: false, frames: [], missing: 0 });
  const predRef = useRef(null);
  const inFlightRef = useRef(false);
  const genRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [trackerReady, setTrackerReady] = useState(false);
  const [handsCount, setHandsCount] = useState(0);
  const [live, setLive] = useState(null);
  const [lastSign, setLastSign] = useState(null);
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
      clearInterval(timerRef.current);
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
        landmarkerRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // The hand left the frame: make one final guess for the whole sign
  const finalizeSegment = async () => {
    const seg = segRef.current;
    segRef.current = { active: false, frames: [], missing: 0 };
    const frames = seg.frames.slice(0, Math.max(seg.frames.length - seg.missing, 0));
    const gen = ++genRef.current;
    setLive(null);
    predRef.current = null;
    if (frames.filter(hasHand).length < MIN_HAND_FRAMES) return;
    try {
      const data = await requestPrediction(frames);
      if (gen !== genRef.current || !data.label) return;
      setLastSign(data);
      predRef.current =
        data.confidence >= MIN_CONFIDENCE
          ? { ...data, at: performance.now(), ttl: HOLD_FINAL_MS }
          : null;
    } catch (err) {
      console.error(err);
    }
  };

  // Every video frame: detect hands, draw, and collect the sign so far
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
        const frame = resultToFrame(result);
        const present = hasHand(frame);
        const seg = segRef.current;

        if (present) {
          if (!seg.active) {
            seg.active = true;
            seg.frames = [];
          }
          seg.missing = 0;
          seg.frames.push(frame);
        } else if (seg.active) {
          seg.missing += 1;
          seg.frames.push(frame);
          if (seg.missing >= END_AFTER_MISSING) finalizeSegment();
        }
        if (seg.frames.length > MAX_SEGMENT_FRAMES) seg.frames.shift();

        drawHands(canvas.getContext("2d"), result.landmarks, canvas.width, canvas.height, predRef.current);
        setHandsCount(result.landmarks.length);
      }
    }
    rafRef.current = requestAnimationFrame(detectLoop);
  };

  // Every 0.4 s while a sign is in progress: guess from everything collected so far
  const predictTick = async () => {
    const seg = segRef.current;
    if (!seg.active || inFlightRef.current) return;
    const frames = seg.frames.slice(0, Math.max(seg.frames.length - seg.missing, 0));
    if (frames.filter(hasHand).length < MIN_HAND_FRAMES) return;

    inFlightRef.current = true;
    const gen = genRef.current;
    try {
      const data = await requestPrediction(frames);
      if (gen !== genRef.current) return;
      if (data.label) {
        setLive(data);
        predRef.current =
          data.confidence >= MIN_CONFIDENCE
            ? { ...data, at: performance.now(), ttl: 1200 }
            : null;
      }
    } catch (err) {
      console.error(err);
    } finally {
      inFlightRef.current = false;
    }
  };

  const stopCamera = () => {
    cancelAnimationFrame(rafRef.current);
    clearInterval(timerRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    segRef.current = { active: false, frames: [], missing: 0 };
    predRef.current = null;
    genRef.current += 1;
    setLive(null);
    setLastSign(null);
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
      segRef.current = { active: false, frames: [], missing: 0 };
      rafRef.current = requestAnimationFrame(detectLoop);
      timerRef.current = setInterval(predictTick, PREDICT_EVERY_MS);
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

      {running && (
        <p className="hand-status">
          Hands detected: {handsCount}
          {live &&
            `  |  Live guess: ${live.top3
              .map((t) => `${t.label} ${(t.confidence * 100).toFixed(0)}%`)
              .join(", ")}`}
        </p>
      )}

      {lastSign && (
        <div className="result-card">
          <div className="result-main">
            Last sign: {lastSign.label} <span>{(lastSign.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="result-top3">
            {lastSign.top3.map((t, i) => (
              <span key={t.label}>
                {i + 1}. {t.label} {(t.confidence * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      )}
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}