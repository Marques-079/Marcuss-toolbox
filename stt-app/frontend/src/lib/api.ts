import axios from "axios";

// Create an Axios client that talks to your backend FastAPI server
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

// Upload audio blob to /transcribe
export async function uploadAudio(blob: Blob) {
  const form = new FormData();
  form.append("file", blob, "recording.webm");

  const res = await api.post("/transcribe", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return res.data as {
    language: string;
    duration: number;
    text: string;
    segments: { start: number; end: number; text: string }[];
  };
}

// Check if backend is alive
export async function health() {
  const res = await api.get("/health");
  return res.data;
}
