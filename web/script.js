/* Original JavaScript — unchanged */
const selectBtn = document.getElementById("selectBtn");
const urlBtn = document.getElementById("urlBtn");
const videoURL = document.getElementById("videoURL");
const statusEl = document.getElementById("status");
const videoPathEl = document.getElementById("videoPath");
const audioPathEl = document.getElementById("audioPath");
const transcriptEl = document.getElementById("transcript");
const copyBtn = document.getElementById("copyBtn");
const saveBtn = document.getElementById("saveBtn");

async function fetchJSON(url, method="GET", body=null) {
  let opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res = await fetch(url, opts);
  return await res.json();
}

selectBtn.addEventListener("click", async () => {
  statusEl.innerText = "Processing...";
  transcriptEl.innerText = "Processing...";

  const result = await fetchJSON("/select_video");

  if (result.error) {
    statusEl.innerText = "Error";
    transcriptEl.innerText = result.error;
    return;
  }

  videoPathEl.innerText = result.video_path;
  audioPathEl.innerText = result.audio_path;
  transcriptEl.innerText = result.text;
  statusEl.innerText = "Done";
});

/* ⭐ NEW — URL Processing */
urlBtn.addEventListener("click", async () => {
  const url = videoURL.value.trim();
  if (!url) return alert("Enter a valid link");

  statusEl.innerText = "Downloading...";
  transcriptEl.innerText = "Downloading and processing...";

  const result = await fetchJSON("/url_to_text", "POST", { url });

  if (result.error) {
    statusEl.innerText = "Error";
    transcriptEl.innerText = result.error;
    return;
  }

  videoPathEl.innerText = result.video_source;
  audioPathEl.innerText = "(auto-deleted)";
  transcriptEl.innerText = result.text;
  statusEl.innerText = "Done";
});

/* Copy & Save */
copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(transcriptEl.innerText);
  alert("Copied transcript");
});
saveBtn.addEventListener("click", () => {
  const text = transcriptEl.innerText;
  const blob = new Blob([text], {type: "text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "transcript.txt";
  a.click();
});
document.getElementById("clearLocalVideoBtn").onclick = async () => {
    const res = await fetch("/delete_local_video");
    const data = await res.json();
    alert(data.message);
};