const mediaInput = document.getElementById("media");

const detectButton = document.getElementById("detectBtn");

const loading = document.getElementById("loading");
const livePreview = document.getElementById("livePreview");

const resultContainer = document.getElementById("resultContainer");

const WS_BASE = "ws://127.0.0.1:8000/ws/detect";
const UPLOAD_URL = "http://127.0.0.1:8000/upload";

let foundCount = 0;

detectButton.addEventListener("click", async function () {
    const file = mediaInput.files[0];

    if (!file) {
        alert("please select an image or video");
        return;
    }

    foundCount = 0;
    loading.style.display = "block";
    loading.textContent = "Uploading";
    resultContainer.innerHTML = "";

    let fileId;

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(UPLOAD_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Upload failed");
        }

        fileId = data.file_id;
    } catch (error) {
        loading.style.display = "none";
        showError(error.message);
        return;
    }

    loading.textContent = "Processing";
    livePreview.style.display = "block";

    const socket = new WebSocket(`${WS_BASE}/${fileId}`);

    socket.onmessage = function (event) {
        const update = JSON.parse(event.data);

        if (update.type === "frame") {
            livePreview.src = `data:image/jpeg;base64,${update.image}`;
        } else if (update.type === "plate_found") {
            appendResultCard(update);
            foundCount++;
            loading.textContent = `Processing (${foundCount} found)`;
        } else if (update.type === "done") {
            loading.style.display = "none";
            if (foundCount === 0) {
                resultContainer.innerHTML =
                    "<p class='no-result'>No number plates detected.</p>";
            }
            socket.close();
        } else if (update.type === "error") {
            loading.style.display = "none";
            showError(update.message || "Detection failed");
            socket.close();
        }
    };

    socket.onerror = function () {
        loading.style.display = "none";
        showError("Connection error while talking to the server.");
    };
});

function showError(message) {
    resultContainer.innerHTML = `
        <p class="error-text">
            Error: ${message}
        </p>
    `;
}

function appendResultCard(result) {
    const card = document.createElement("div");
    card.className = "result-card";

    const hasConfidence = typeof result.confidence === "number";
    const confidencePct = hasConfidence ? Math.round(result.confidence * 100) : null;
    const confidenceClass = hasConfidence && confidencePct < 60 ? "confidence low" : "confidence";

    card.innerHTML = `
             <span class="vehicle-label">${result.vehicle}</span>
    <div class="plate-block">
        ${result.plate_image
            ? `<img class="plate-img" src="data:image/jpeg;base64,${result.plate_image}" alt="${result.plate}">`
            : ""}
        <span class="plate-badge">${result.plate}</span>
    </div>
    <span class="${confidenceClass}">${hasConfidence ? confidencePct + "%" : "—"}</span>
       
    `;

    resultContainer.appendChild(card);
}