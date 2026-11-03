import { useEffect, useState } from "react";
import WebcamView from "./components/WebcamView";

function App() {
  const [status, setStatus] = useState("checking...");

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
        <span className={`status-pill ${status === "ok" ? "ok" : "bad"}`}>
          Backend: {status}
        </span>
      </header>

      <main>
        <WebcamView />
      </main>
    </div>
  );
}

export default App;