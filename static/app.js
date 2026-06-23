// AuraShop Support Chat & CV Pipeline Controller

// Generate a random session ID on page load
const sessionId = 'session_' + Math.random().toString(36).substring(2, 11);

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const quickRepliesContainer = document.getElementById('quick-replies-container');
const uploadOverlay = document.getElementById('upload-overlay');
const cancelUploadBtn = document.getElementById('cancel-upload-btn');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const resetChatBtn = document.getElementById('reset-chat-btn');
const dialogStateBadge = document.getElementById('dialog-state-badge');

// Vision Panel Elements
const cvEmptyState = document.getElementById('cv-empty-state');
const cvActiveDisplay = document.getElementById('cv-active-display');
const cvPreviewImg = document.getElementById('cv-preview-img');
const cvProcessedCanvas = document.getElementById('cv-processed-canvas');
const statContrast = document.getElementById('stat-contrast');
const statEdges = document.getElementById('stat-edges');
const statBrightness = document.getElementById('stat-brightness');
const statTensor = document.getElementById('stat-tensor');
const verdictBanner = document.getElementById('verdict-banner');
const verdictTag = document.getElementById('verdict-tag');
const confidencePercentage = document.getElementById('confidence-percentage');
const confidenceFill = document.getElementById('confidence-fill');
const verdictDesc = document.getElementById('verdict-desc');
const pipelineLogs = document.getElementById('pipeline-logs');

// Quick Sandbox Simulation Buttons
const simDefectBtn = document.getElementById('sim-defect-btn');
const simCleanBtn = document.getElementById('sim-clean-btn');

// Initialize Chat
document.addEventListener('DOMContentLoaded', () => {
    // Send empty greeting message or trigger bot greeting
    appendBotMessage("Hello! I am your AuraShop Support Assistant. How can I help you today? You can track an order, ask about returns, or file an exchange for a defective item.");
    renderQuickReplies(["Track Order", "Return Policy", "File Return / Exchange", "Talk to Agent"]);
    updateDialogState("GREETING");
});

// Reset Chat Event
resetChatBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        if (response.ok) {
            chatMessages.innerHTML = '';
            appendBotMessage("Hi there! I'm your E-commerce Support Bot. How can I help you today?");
            renderQuickReplies(["Track Order", "Return Policy", "File Return / Exchange", "Talk to Agent"]);
            updateDialogState("GREETING");
            hideUploadZone();
            resetVisionPanel();
        }
    } catch (err) {
        console.error("Error resetting session:", err);
    }
});

// Form Submission Event
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;
    
    chatInput.value = '';
    await handleUserMessage(text);
});

// Dismiss Upload Trigger
cancelUploadBtn.addEventListener('click', () => {
    hideUploadZone();
    appendBotMessage("Upload cancelled. Returning to general support chat. What else can I assist you with?");
    renderQuickReplies(["Track Order", "Return Policy", "File Return / Exchange", "Talk to Agent"]);
    updateDialogState("CHATTING");
});

// Trigger File Selector on Click
dropzone.addEventListener('click', () => {
    fileInput.click();
});

// File Selection Event
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
});

// Drag & Drop Setup
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('border-accent-cyan', 'bg-slate-950/80');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('border-accent-cyan', 'bg-slate-950/80');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('border-accent-cyan', 'bg-slate-950/80');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        uploadFile(file);
    }
});

// Sandbox Simulator Button Events
simDefectBtn.addEventListener('click', () => generateAndUploadMockImage(true));
simCleanBtn.addEventListener('click', () => generateAndUploadMockImage(false));

/**
 * Handles sending messages to the chatbot and updating UI.
 */
async function handleUserMessage(messageText) {
    appendUserMessage(messageText);
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: messageText })
        });
        
        removeTypingIndicator();
        
        if (response.ok) {
            const data = await response.json();
            appendBotMessage(data.response);
            renderQuickReplies(data.quick_replies);
            updateDialogState(data.state);
            
            if (data.prompt_upload) {
                showUploadZone();
            } else {
                hideUploadZone();
            }
        } else {
            appendBotMessage("Sorry, I encountered an error communicating with the support agent pipeline. Please try again.");
        }
    } catch (err) {
        removeTypingIndicator();
        console.error("Chat error:", err);
        appendBotMessage("Connection failed. Please ensure the development backend server is running.");
    }
}

/**
 * Prepares and sends an image file to the `/api/upload` endpoint.
 */
async function uploadFile(file) {
    // Show local preview instantly on the right panel
    const reader = new FileReader();
    reader.onload = (e) => {
        setupActiveVisionPanel(e.target.result);
        simulateProcessedImage(e.target.result);
    };
    reader.readAsDataURL(file);
    
    // UI state updates: show loading/processing
    showTypingIndicator();
    clearPipelineLogs();
    appendLog("⏳ Initializing image transfer...", "info");
    
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        removeTypingIndicator();
        
        if (response.ok) {
            const data = await response.json();
            
            // Render backend results in visual panels
            renderClassificationResults(data);
            
            // Add bot's feedback message in the chat
            appendBotMessage(data.bot_response);
            renderQuickReplies(data.quick_replies);
            updateDialogState(data.next_state);
            hideUploadZone();
        } else {
            const errDetails = await response.json();
            appendBotMessage(`❌ Upload processing failed: ${errDetails.detail || 'Unknown error'}`);
            appendLog(`❌ Execution aborted: ${errDetails.detail || 'Internal error'}`, "error");
        }
    } catch (err) {
        removeTypingIndicator();
        console.error("Upload error:", err);
        appendBotMessage("❌ Failed to upload image to the CV endpoint. Check connection.");
        appendLog("❌ HTTP connection error", "error");
    }
}

/**
 * Generates a mock image locally using HTML5 canvas and uploads it.
 * This simulates uploading a defective camera or a clean one.
 */
function generateAndUploadMockImage(isDefective) {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    const ctx = canvas.getContext('2d');
    
    // Draw background/product casing
    const grad = ctx.createRadialGradient(200, 200, 50, 200, 200, 180);
    grad.addColorStop(0, '#334155');
    grad.addColorStop(1, '#0f172a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 400, 400);
    
    // Draw lens/casing detail
    ctx.beginPath();
    ctx.arc(200, 200, 100, 0, 2 * Math.PI);
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 6;
    ctx.fill();
    ctx.stroke();
    
    // Glass reflex
    ctx.beginPath();
    ctx.arc(200, 200, 80, 0, 2 * Math.PI);
    ctx.fillStyle = '#0f172a';
    ctx.fill();
    
    if (isDefective) {
        // Draw jagged red/dark crack lines across the lens
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(140, 150);
        ctx.lineTo(210, 190);
        ctx.lineTo(200, 220);
        ctx.lineTo(260, 270);
        ctx.stroke();
        
        // Additional fracture branches
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(210, 190);
        ctx.lineTo(230, 170);
        ctx.moveTo(200, 220);
        ctx.lineTo(170, 240);
        ctx.stroke();
        
        ctx.fillStyle = '#f43f5e';
        ctx.fillText("CRACK DETECTED", 20, 30);
    } else {
        // Draw a clean camera lens reflection
        ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
        ctx.beginPath();
        ctx.ellipse(220, 180, 40, 20, Math.PI / 4, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    const filename = isDefective ? "defect_camera_shattered_lens.jpg" : "clean_product_casing.jpg";
    
    canvas.toBlob((blob) => {
        const file = new File([blob], filename, { type: 'image/jpeg' });
        uploadFile(file);
    }, 'image/jpeg');
}

/**
 * UI Rendering Helpers
 */

function appendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex justify-end chat-bubble-animate';
    msgDiv.innerHTML = `
        <div class="bg-blue-600 text-white text-sm max-w-[80%] rounded-lg rounded-tr-none px-4 py-2.5 shadow-sm">
            ${escapeHtml(text)}
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    scrollChatToBottom();
}

function appendBotMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'flex justify-start chat-bubble-animate';
    
    // Simple markdown-to-html conversion for bold and lists
    let formattedText = escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/•\s(.*?)(<br>|$)/g, '<li>$1</li>');
        
    if (formattedText.includes('<li>')) {
        formattedText = formattedText.replace(/(<li>.*?<\/li>)/g, '<ul class="list-disc pl-4 space-y-1 my-1">$1</ul>');
    }

    msgDiv.innerHTML = `
        <div class="flex gap-2 max-w-[85%]">
            <div class="text-xl flex-shrink-0">
                🤖
            </div>
            <div class="bg-gray-100 border border-gray-200 text-gray-800 text-sm rounded-lg rounded-tl-none px-4 py-2.5 shadow-sm leading-relaxed">
                ${formattedText}
            </div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    scrollChatToBottom();
}

function showTypingIndicator() {
    removeTypingIndicator(); // Ensure no duplicates
    const indicator = document.createElement('div');
    indicator.id = 'typing-indicator';
    indicator.className = 'flex justify-start chat-bubble-animate';
    indicator.innerHTML = `
        <div class="flex gap-2">
            <div class="text-xl flex-shrink-0">
                🤖
            </div>
            <div class="bg-gray-100 border border-gray-200 text-gray-400 rounded-lg rounded-tl-none px-4 py-3 flex items-center gap-1 shadow-sm">
                <span class="typing-dot h-1.5 w-1.5 rounded-full"></span>
                <span class="typing-dot h-1.5 w-1.5 rounded-full"></span>
                <span class="typing-dot h-1.5 w-1.5 rounded-full"></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    scrollChatToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function renderQuickReplies(replies) {
    quickRepliesContainer.innerHTML = '';
    if (!replies || replies.length === 0) return;
    
    replies.forEach(reply => {
        const btn = document.createElement('button');
        btn.className = 'text-xs font-semibold px-3 py-1.5 bg-white hover:bg-gray-100 border border-gray-300 text-blue-600 rounded shadow-sm transition-colors';
        btn.textContent = reply;
        btn.addEventListener('click', () => handleUserMessage(reply));
        quickRepliesContainer.appendChild(btn);
    });
}

function showUploadZone() {
    uploadOverlay.classList.remove('hidden');
}

function hideUploadZone() {
    uploadOverlay.classList.add('hidden');
}

function updateDialogState(state) {
    dialogStateBadge.textContent = `STATE: ${state}`;
}

function scrollChatToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Computer Vision Monitor Panel Updates
 */

function setupActiveVisionPanel(imgDataUrl) {
    cvEmptyState.classList.add('hidden');
    cvActiveDisplay.classList.remove('hidden');
    cvPreviewImg.src = imgDataUrl;
}

function resetVisionPanel() {
    cvEmptyState.classList.remove('hidden');
    cvActiveDisplay.classList.add('hidden');
    cvPreviewImg.src = '';
    clearPipelineLogs();
}

function simulateProcessedImage(imgDataUrl) {
    const img = new Image();
    img.src = imgDataUrl;
    img.onload = () => {
        const ctx = cvProcessedCanvas.getContext('2d');
        // Clear canvas
        ctx.clearRect(0, 0, 224, 224);
        
        // Draw image downscaled to 224x224
        ctx.drawImage(img, 0, 0, 224, 224);
        
        // Apply a visual grid pixelation filter and convert to grayscale to simulate raw tensor analysis
        const imgData = ctx.getImageData(0, 0, 224, 224);
        const data = imgData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const gray = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
            // Standard clean grayscale
            data[i] = gray;     // Red
            data[i+1] = gray;   // Green
            data[i+2] = gray;   // Blue
        }
        ctx.putImageData(imgData, 0, 0);
    };
}

function clearPipelineLogs() {
    pipelineLogs.innerHTML = '';
}

function appendLog(message, type = "info") {
    const logDiv = document.createElement('div');
    logDiv.className = 'log-entry py-0.5 border-b border-gray-800/40';
    
    let colorClass = 'text-gray-400';
    let prefix = '[-]';
    
    if (type === "success") {
        colorClass = 'text-green-400';
        prefix = '[+]';
    } else if (type === "error") {
        colorClass = 'text-red-400';
        prefix = '[!]';
    } else if (type === "warning") {
        colorClass = 'text-yellow-400';
        prefix = '[*]';
    } else if (type === "info") {
        colorClass = 'text-blue-400';
        prefix = '[~]';
    }
    
    logDiv.innerHTML = `<span class="${colorClass} font-bold mr-1.5">${prefix}</span> <span class="text-green-300">${escapeHtml(message)}</span>`;
    pipelineLogs.appendChild(logDiv);
    pipelineLogs.scrollTop = pipelineLogs.scrollHeight;
}

function renderClassificationResults(data) {
    // 1. Update stats
    statContrast.textContent = data.stats.contrast.toFixed(4);
    statEdges.textContent = data.stats.edge_density.toFixed(4);
    statBrightness.textContent = data.stats.brightness.toFixed(4);
    statTensor.textContent = `[${data.stats.tensor_shape.join(',')}]`;
    
    // 2. Animate confidence bar
    const conf = data.confidence;
    const confPercent = (conf * 100).toFixed(1) + '%';
    confidencePercentage.textContent = confPercent;
    
    confidenceFill.style.width = confPercent;
    
    // 3. Style verdict banner depending on result
    verdictBanner.className = 'rounded p-3.5 border flex flex-col gap-2 transition-all duration-300';
    verdictTag.textContent = data.verdict.toUpperCase();
    verdictDesc.textContent = data.details;
    
    if (data.verified) {
        // Defect Verified
        verdictBanner.classList.add('bg-red-50', 'border-red-200', 'text-red-900');
        verdictTag.className = 'text-[10px] font-bold px-2 py-0.5 rounded bg-red-100 text-red-700 border border-red-200';
        confidenceFill.className = 'h-full rounded bg-red-600';
    } else {
        // Defect Rejected
        verdictBanner.classList.add('bg-green-50', 'border-green-200', 'text-green-900');
        verdictTag.className = 'text-[10px] font-bold px-2 py-0.5 rounded bg-green-100 text-green-700 border border-green-200';
        confidenceFill.className = 'h-full rounded bg-green-600';
    }
    
    // 4. Sequentially render logs to feel alive
    let delay = 0;
    data.logs.forEach((logText) => {
        setTimeout(() => {
            let logType = "info";
            if (logText.includes("Loaded") || logText.includes("Resized") || logText.includes("Scaled") || logText.includes("normalization")) {
                logType = "success";
            } else if (logText.includes("completed")) {
                logType = "success";
            }
            appendLog(logText, logType);
        }, delay);
        delay += 150; // 150ms between each log statement
    });
    
    // Final output log after all preprocessing logs
    setTimeout(() => {
        if (data.verified) {
            appendLog(`Model Verdict: DEFECT_DETECTED (Confidence: ${confPercent})`, "warning");
            appendLog("Executing return routing loop...", "info");
        } else {
            appendLog(`Model Verdict: NORMAL_STATUS (Confidence: ${confPercent})`, "success");
            appendLog("Triggering chatbot rejection feedback...", "info");
        }
    }, delay);
}

// Utility to escape HTML and prevent injection attacks
function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
