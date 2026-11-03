import { useEffect, useRef, useState } from "react";

export default function WebcamView() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
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
    } catch (err) {
      setError("Could not access the camera. Please allow camera permission and try again.");
    }
  };

  // Turn the camera off if the user leaves this page
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="webcam-card">
      <div className="video-frame">
        <video ref={videoRef} className="video" playsInline muted />
        {!running && <div className="video-placeholder">Camera is off</div>}
      </div>

      <div className="webcam-actions">
        {!running ? (
          <button className="btn btn-primary" onClick={startCamera}>
            Start camera
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopCamera}>
            Stop camera
          </button>
        )}
      </div>

      {error && <p className="error-text">{error}</p>}
    </div>
  );
}