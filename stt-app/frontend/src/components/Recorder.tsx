import React, { useRef, useState } from "react";
import { uploadAudio, health } from "../lib/api";

function sToTimestamp(s: number) {
  const mm = Math.floor(s / 60).toString().padStart(2, "0");
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${mm}:${ss}`;
}

export default function Recorder() {
  const [status, setStatus] = useState<"idle" | "recording" | "uploading" | "done" | "error">("idle");
  const [transcript, setTranscript] = useState<string>("");
  const [segments, setSegments] = useState<{ start: number; end: number; text: string }[]>([]);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const [info, setInfo] = useState<string>("");

  async function checkBackend() {
    try {
      const h = await health();
      setInfo(`API: ${JSON.stringify(h)}`);
    } catch (e) {
      setInfo("Cannot reach API. Is backend running on 8000?");
    }
  }

  async function start() {
    setTranscript("");
    setSegments([]);
    setStatus("recording");

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Prefer webm/opus (Chrome/Edge). Browser will fall back automatically.
    const preferred = "audio/webm;codecs=opus";
    const options = MediaRecorder.isTypeSupported(preferred) ? { mimeType: preferred } : undefined;

    const mr = new MediaRecorder(stream, options as MediaRecorderOptions);
    chunksRef.current = [];

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = async () => {
      setStatus("uploading");
      const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
      try {
        const result = await uploadAudio(blob);
        setTranscript(result.text);
        setSegments(result.segments);
        setStatus("done");
      } catch (err: any) {
        console.error(err);
        setTranscript("");
        setSegments([]);
        setStatus("error");
      }
    };

    recRef.current = mr;
    mr.start();
  }

  function stop() {
    if (recRef.current && recRef.current.state !== "inactive") {
      recRef.current.stop();
      // also stop the mic tracks to release the device
      recRef.current.stream.getTracks().forEach((t) => t.stop());
    }
  }

  async function uploadFileFromDisk(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("uploading");
    try {
      const result = await uploadAudio(file);
      setTranscript(result.text);
      setSegments(result.segments);
      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-2">🎤 Faster‑Whisper — Speech to Text</h1>
      <p className="mb-4 text-sm text-gray-600">Record or upload audio; the server transcribes it with faster‑whisper.</p>

      <div className="flex gap-2 mb-4">
        <button onClick={start} disabled={status === "recording"} className="px-4 py-2 rounded bg-black text-white disabled:opacity-50">Start</button>
        <button onClick={stop} disabled={status !== "recording"} className="px-4 py-2 rounded bg-gray-800 text-white disabled:opacity-50">Stop</button>
        <label className="px-4 py-2 rounded bg-gray-200 cursor-pointer">
          Upload file
          <input type="file" accept="audio/*,video/*" className="hidden" onChange={uploadFileFromDisk} />
        </label>
        <button onClick={checkBackend} className="px-4 py-2 rounded border">Check API</button>
      </div>

      <div className="mb-4 text-sm">Status: <b>{status}</b> {info && <span className="ml-2">({info})</span>}</div>

      {transcript && (
        <>
          <h2 className="text-xl font-semibold mt-4 mb-2">Transcript</h2>
          <div className="whitespace-pre-wrap p-3 rounded border">{transcript}</div>

          <h3 className="text-lg font-semibold mt-4 mb-2">Segments</h3>
          <table className="w-full text-sm border">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 border">Start</th>
                <th className="p-2 border">End</th>
                <th className="p-2 border">Text</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((s, i) => (
                <tr key={i}>
                  <td className="p-2 border">{sToTimestamp(s.start)}</td>
                  <td className="p-2 border">{sToTimestamp(s.end)}</td>
                  <td className="p-2 border">{s.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {status === "error" && (
        <div className="mt-4 p-3 rounded bg-red-100 text-red-800">Something went wrong. Make sure the backend is running and FFmpeg is installed.</div>
      )}
    </div>
  );
}
