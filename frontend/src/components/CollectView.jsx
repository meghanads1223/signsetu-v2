import { useEffect, useRef, useState } from "react";
import { FilesetResolver } from "@mediapipe/tasks-vision";
import { API_URL, createLandmarker, drawSkeleton, hasHand, resultToFrame } from "../lib/hands";

const END_AFTER_MISSING = 12;  // frames without a hand that end a take (about 0.4 s)
const MIN_HAND_FRAMES = 12;    // shorter takes are discarded
const MAX_TAKE_FRAMES = 180;   // about 6 seconds, safety limit
const TARGET_TAKES = 10;

export default function CollectView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(-1);
  const segRef = useRef({ active: false, frames: [], missing: 0 });
  const collectingRef = useRef(false);
  const labelRef = useRef("");

  const [signs, setSigns] = useState([]);
  const [label, setLabel] = useState("");
  const [running, setRunning] = useState(false);
  const [trackerReady, setTrackerReady] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [handsCount, setHandsCount] = useState(0);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");

  const setCount = (lab, count) =>
    setSigns((prev) => prev.map((s) => (s.label === lab ? { ...s, count } : s)));

  // Load the list of signs from the backend
  useEffect(() => {
    fetch(`${API_URL}/collect/signs`)
      .then((r) => r.json())
      .then((data) => {
        setSigns(data.signs);
        if (data.signs.length) {
          setLabel(data.signs[0].label);
          labelRef.current = data.signs[0].label;
        }
      })
      .catch(() => setError("Could not reach the backend. Is it running?"));
  }, []);

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
        if (!cancelled) setError("Could not load the hand tracker.");
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

  const saveTake = async (frames, lab) => {
    if (frames.filter(hasHand).length < MIN_HAND_FRAMES) {
      setStatus({ type: "warn", text: "That take was too short and was discarded. Sign again." });
      return;
    }
    try {
      const res = await fetch(`${API_URL}/collect/take`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: lab, frames }),
      });
      if (!res.ok) throw new Error("save failed");
      const data = await res.json();
      setCount(lab, data.count);
      setStatus({ type: "ok", text: `Saved take ${data.count} for "${lab}"` });
    } catch (err) {
      console.error(err);
      setStatus({ type: "bad", text: "Could not save the take. Is the backend running?" });
    }
  };

  const finalizeSegment = () => {
    const seg = segRef.current;
    segRef.current = { active: false, frames: [], missing: 0 };
    if (!collectingRef.current) return;
    const frames = seg.frames.slice(0, Math.max(seg.frames.length - seg.missing, 0));
    saveTake(frames, labelRef.current);
  };

  // Every video frame: detect hands, draw, and collect the current take
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
        const seg = segRef.current;

        if (hasHand(frame)) {
          if (!seg.active) {
            seg.active = true;
            seg.frames = [];
          }
          seg.missing = 0;
          seg.frames.push(frame);
          if (seg.frames.length >= MAX_TAKE_FRAMES) finalizeSegment();
        } else if (seg.active) {
          seg.missing += 1;
          seg.frames.push(frame);
          if (seg.missing >= END_AFTER_MISSING) finalizeSegment();
        }

        const text = collectingRef.current ? (segRef.current.active ? "Recording..." : "Ready") : "Paused";
        drawSkeleton(canvas.getContext("2d"), result.landmarks, canvas.width, canvas.height, [text]);
        setHandsCount(result.landmarks.length);
      }
    }
    rafRef.current = requestAnimationFrame(detectLoop);
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
    } catch (err) {
      setError("Could not access the camera. Please allow camera permission and try again.");
    }
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
    segRef.current = { active: false, frames: [], missing: 0 };
    collectingRef.current = false;
    setCollecting(false);
    setHandsCount(0);
    setRunning(false);
  };

  const toggleCollecting = () => {
    const next = !collectingRef.current;
    collectingRef.current = next;
    segRef.current = { active: false, frames: [], missing: 0 };
    setCollecting(next);
    setStatus(null);
  };

  const onSignChange = (e) => {
    setLabel(e.target.value);
    labelRef.current = e.target.value;
    segRef.current = { active: false, frames: [], missing: 0 };
    setStatus(null);
  };

  const undoLast = async () => {
    try {
      const res = await fetch(`${API_URL}/collect/take?label=${encodeURIComponent(label)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      setCount(label, data.count);
      setStatus({ type: "ok", text: `Removed the last take. "${label}" now has ${data.count}.` });
    } catch (err) {
      setStatus({ type: "bad", text: "Could not remove the take." });
    }
  };

  const current = signs.find((s) => s.label === label);
  const count = current ? current.count : 0;
  const total = signs.reduce((sum, s) => sum + s.count, 0);

  return (
    <div className="webcam-card">
      <div className="collect-top">
        <label>
          Sign to record{" "}
          <select value={label} onChange={onSignChange} className="sign-select">
            {signs.map((s) => (
              <option key={s.label} value={s.label}>
                {s.label} ({s.count} takes)
              </option>
            ))}
          </select>
        </label>
        <span className="hand-status">All signs: {total} takes saved</span>
      </div>

      <div className="video-frame">
        <video ref={videoRef} className="video" playsInline muted />
        <canvas ref={canvasRef} className="overlay" />
        {!running && <div className="video-placeholder">Camera is off</div>}
        {collecting && <div className="rec-badge">● Collecting "{label}"</div>}
      </div>

      <div className="progress">
        <div className="progress-bar" style={{ width: `${Math.min(count / TARGET_TAKES, 1) * 100}%` }} />
      </div>
      <p className="hand-status">
        {count} of {TARGET_TAKES} takes for "{label}" | Hands detected: {handsCount}
      </p>

      <div className="webcam-actions">
        {!running ? (
          <button className="btn btn-primary" onClick={startCamera} disabled={!trackerReady}>
            {trackerReady ? "Start camera" : "Loading hand tracker..."}
          </button>
        ) : (
          <>
            <button className={`btn ${collecting ? "btn-recording" : "btn-record"}`} onClick={toggleCollecting}>
              {collecting ? "Pause collecting" : "Start collecting"}
            </button>
            <button className="btn btn-secondary" onClick={undoLast}>
              Undo last take
            </button>
            <button className="btn btn-danger" onClick={stopCamera}>
              Stop camera
            </button>
          </>
        )}
      </div>

      {status && <p className={`status-line ${status.type}`}>{status.text}</p>}
      {error && <p className="error-text">{error}</p>}

      <p className="hint-text">
        Tip: hold your hand out of view, do the whole sign at normal speed, then drop your hand. Each time your hand
        leaves the frame, one take is saved automatically.
      </p>
    </div>
  );
}