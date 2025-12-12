const API_BASE = "http://127.0.0.1:8081";

// DOM elements
// various UI components
const modelSelect = document.getElementById('model-select');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
// various model parameters
const tempInput = document.getElementById('temperature');
const topPInput = document.getElementById('top-p');
const maxTokensInput = document.getElementById('max-tokens');
const seedInput = document.getElementById('seed');
// Update the number display next to the sliders in real-time
tempInput.oninput = (e) => document.getElementById('temp-val').textContent = e.target.value;
topPInput.oninput = (e) => document.getElementById('topp-val').textContent = e.target.value;

let isGenerating = false;
let activeModel = "";

// Called when page loads
async function init() {
    try {
        // Health check to see if server is running
        const health = await fetch(`${API_BASE}/health`);
        if (health.ok) {
            statusDot.classList.add('online');
            statusText.textContent = "Server Online";
        }

        // Get the list of models
        const res = await fetch(`${API_BASE}/models`);
        const data = await res.json(); 
        modelSelect.innerHTML = "";
        activeModel = data.active;

        // Populate model dropdown
        Object.keys(data.models).forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name.toUpperCase();
            // Select the model if it's already loaded
            if (name === activeModel) option.selected = true;
            modelSelect.appendChild(option);
        });
        // Load model when user select
        modelSelect.addEventListener('change', (e) => loadModel(e.target.value));
    } catch (e) {
        statusText.textContent = "Server Offline";
        console.error(e);
    }
}

// Load model
async function loadModel(modelName) {
    if (!modelName) return;
    // Show loading ... on page
    loadingOverlay.style.display = 'flex';
    loadingText.textContent = `Loading ${modelName}...`;
    try {
        const res = await fetch(`${API_BASE}/load_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: modelName })
        });
        
        const data = await res.json();
        if (data.status === 'ok') {
            activeModel = modelName;
            // Notify user when model is loaded
            addMessage("System", `Model loaded: ${modelName}`, "ai");
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (e) {
        alert("Failed to connect to server.");
    } finally {
        // Hide loading overlay
        loadingOverlay.style.display = 'none';
    }
}

// Create new message box and append to chat history
function addMessage(role, text, type) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;// user or ai type
    msgDiv.innerHTML = `
        <div class="avatar">${role === 'User' ? 'U' : 'AI'}</div>
        <div class="content">${text}</div>
    `;  
    chatHistory.appendChild(msgDiv);
    // Auto scroll to botton for new messages
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv.querySelector('.content'); 
}

// Send messages token by token
async function sendMessage() {
    const text = userInput.value.trim();
    // do not send empty messages or already generating
    if (!text || isGenerating) return;
    // Auto load when no model is selected
    if (!activeModel && modelSelect.value) {
        await loadModel(modelSelect.value);
    }
    if (!activeModel) {
        alert("Please select a model first.");
        return;
    }
    // clear user input and show user message
    userInput.value = "";
    userInput.style.height = '50px';
    addMessage("User", text, "user");
    
    // Prepare ai message box
    const aiContentDiv = addMessage("AI", "", "ai");
    isGenerating = true;
    sendBtn.disabled = true;
    // Prepare request payload
    const payload = {
        prompt: text,
        temperature: parseFloat(tempInput.value),
        top_p: parseFloat(topPInput.value),
        max_tokens: parseInt(maxTokensInput.value),
        seed: seedInput.value ? parseInt(seedInput.value) : null
    };

    try {
        // POST /infer_stream
        const response = await fetch(`${API_BASE}/infer_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        // Handle stream responses
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            // Decode binary chunk to string and add to buffer
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
            // Split by newlines
            const lines = buffer.split('\n');
            buffer = lines.pop(); 
            for (const line of lines) {
                //Parse SSE format: data: <content>
                // First remove "data" prefix
                if (line.startsWith('data:')) {
                    let content = line.slice(5);
                    // Fix spacing
                    if (content.startsWith(" ")) {
                        content = content.slice(1);
                    }
                    // Check for special signals                
                    if (content === '[DONE]') break;
                    if (content.startsWith('[MODEL:')) continue; 
                    if (content.startsWith('[ERROR]')) {
                        aiContentDiv.textContent += `\nError: ${content}`;
                        break;
                    }
                    // Append text to UI
                    aiContentDiv.textContent += content;
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            }
        }
    } catch (e) {
        aiContentDiv.textContent += "\n[Network Error]";
    } finally {
        // Reset state when done
        isGenerating = false;
        sendBtn.disabled = false;
    }
}

// Event listeners
// Click to send
sendBtn.addEventListener('click', sendMessage);
// Enter to send, Shift+Enter to insert new line
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
// Theme selection
const themeToggleBtn = document.getElementById('theme-toggle');
themeToggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    themeToggleBtn.textContent = isLight ? 'Dark Mode' : 'Light Mode';
});

// Load page
init();