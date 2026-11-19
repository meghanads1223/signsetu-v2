import { useEffect, useState } from "react";
import WebcamView from "./components/WebcamView";
import CollectView from "./components/CollectView";

function App() {
  const [status, setStatus] = useState("checking...");
  const [tab, setTab] = useState("recognize");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/health")
      .then((res) => res.json())
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("not reachable"));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SignSetu</h1>
        <span className={`status-pill ${status === "ok" ? "ok" : "bad"}`}>Backend: {status}</span>
      </header>

      <nav className="tabs">
        <button className={`tab ${tab === "recognize" ? "active" : ""}`} onClick={() => setTab("recognize")}>
          Recognize
        </button>
        <button className={`tab ${tab === "collect" ? "active" : ""}`} onClick={() => setTab("collect")}>
          Collect data
        </button>
      </nav>

      <main>{tab === "recognize" ? <WebcamView /> : <CollectView />}</main>
    </div>
  );
}

export default App;