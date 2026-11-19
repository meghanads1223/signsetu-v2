import { HandLandmarker } from "@mediapipe/tasks-vision";

export const API_URL = "http://127.0.0.1:8000";
export const RED = "#ff2d2d";

export const hasHand = (frame) => frame[0][0][0] !== 0 || frame[1][0][0] !== 0;

// Turns one MediaPipe result into [2 hands][21 points][x, y, z]. A missing hand is all zeros.
export function resultToFrame(result) {
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

export async function createLandmarker(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

// White skeleton lines, red dots, red box, and text lines above the first hand
export function drawSkeleton(ctx, hands, w, h, lines) {
  ctx.clearRect(0, 0, w, h);
  const s = w / 960;

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

    if (handIndex === 0 && lines.length) {
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