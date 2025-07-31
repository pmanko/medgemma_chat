document.addEventListener('DOMContentLoaded', () => {
    // Configuration - can be easily modified for different environments
    const SERVER_URL = 'http://127.0.0.1:3000';
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // milliseconds
    
    // Get references to all the HTML elements we'll need to interact with
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const chatWindow = document.getElementById('chat-window');
    const modelSelect = document.getElementById('model-select');

    // --- Function to add a message to the chat window ---
    function addMessage(sender, messageContent, isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        // Sanitize content to prevent HTML injection
        const p = document.createElement('p');
        p.textContent = messageContent;
        contentDiv.appendChild(p);

        if (isLoading) {
            contentDiv.classList.add('loading-indicator');
            messageDiv.id = 'loading-message';
        }

        messageDiv.appendChild(contentDiv);
        chatWindow.appendChild(messageDiv);

        // Scroll to the bottom of the chat window to see the new message
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // --- Helper function to remove loading message ---
    function removeLoadingMessage() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    // --- Sleep utility for retry delays ---
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // --- Send request with retry logic ---
    async function sendRequestWithRetry(endpoint, prompt) {
        let lastError = null;
        
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: prompt }),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || response.statusText);
                }

                const data = await response.json();
                
                // Success - remove loading and show response
                removeLoadingMessage();
                addMessage('model', data.response);
                return;

            } catch (error) {
                lastError = error;
                console.error(`Attempt ${attempt} failed:`, error);
                
                if (attempt < MAX_RETRIES) {
                    await sleep(RETRY_DELAY * attempt); // Exponential backoff
                }
            }
        }

        // All retries failed
        removeLoadingMessage();
        addMessage('model', `Error after ${MAX_RETRIES} attempts: ${lastError.message}`);
    }

    // --- Handle form submission ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent the default form submission (which reloads the page)

        const prompt = promptInput.value.trim();
        if (!prompt) return; // Don't send empty messages

        const selectedModel = modelSelect.value;
        const endpoint = `${SERVER_URL}/generate/${selectedModel}`;

        // 1. Display the user's message immediately
        addMessage('user', prompt);

        // 2. Clear the input field and show a loading indicator
        promptInput.value = '';
        addMessage('model', 'Thinking...', true);

        // 3. Send the prompt to the backend API with retry logic
        await sendRequestWithRetry(endpoint, prompt);
    });
});
