document.addEventListener('DOMContentLoaded', () => {
    // Configuration - can be easily modified for different environments
    const SERVER_URL = 'http://127.0.0.1:3000';
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // milliseconds
    
    // Get references to all the HTML elements we'll need to interact with
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const chatWindow = document.getElementById('chat-window');
    const currentModelSpan = document.getElementById('current-model');
    const currentPromptSpan = document.getElementById('current-prompt');
    const systemPromptsContainer = document.getElementById('system-prompts');
    
    // Current selections
    let selectedModel = 'phi3';
    let selectedSystemPrompt = 'default';
    let customSystemPrompt = '';

    // System prompt presets
    const systemPrompts = {
        'default': { name: '💬 Default', prompt: '' },
        'helpful': { name: '🤝 Helpful', prompt: 'You are a helpful, harmless, and honest assistant.' },
        'concise': { name: '⚡ Concise', prompt: 'Be brief and direct in your responses.' },
        'medical': { name: '🏥 Medical', prompt: 'You are a medical assistant. Provide educational information only - not medical advice.' },
        'researcher': { name: '🔬 Researcher', prompt: 'Provide evidence-based, well-researched responses.' },
        'custom': { name: '✏️ Custom', prompt: '' }
    };

    // --- Initialize system prompts dropdown ---
    function populateSystemPrompts() {
        systemPromptsContainer.innerHTML = '';
        Object.entries(systemPrompts).forEach(([key, prompt]) => {
            const li = document.createElement('li');
            li.innerHTML = `<a href="#" data-prompt="${key}">${prompt.name}</a>`;
            systemPromptsContainer.appendChild(li);
        });
    }

    // --- Handle dropdown selections ---
    document.addEventListener('click', (e) => {
        if (e.target.matches('[data-model]')) {
            e.preventDefault();
            selectedModel = e.target.dataset.model;
            currentModelSpan.textContent = e.target.textContent;
            
            const dropdown = e.target.closest('details');
            if (dropdown) dropdown.open = false;
        }
        
        if (e.target.matches('[data-prompt]')) {
            e.preventDefault();
            selectedSystemPrompt = e.target.dataset.prompt;
            currentPromptSpan.textContent = systemPrompts[selectedSystemPrompt].name;
            
            // Show custom prompt editor if "Custom" is selected
            if (selectedSystemPrompt === 'custom') {
                showCustomPromptEditor();
            } else {
                hideCustomPromptEditor();
            }
            
            const dropdown = e.target.closest('details');
            if (dropdown) dropdown.open = false;
        }
    });

    // --- Custom prompt editor functions ---
    function showCustomPromptEditor() {
        let editor = document.getElementById('custom-prompt-editor');
        if (!editor) {
            // Template prompt to help users get started
            const templatePrompt = customSystemPrompt || `You are a knowledgeable assistant with expertise in multiple domains. Please:

- Provide accurate, well-researched information
- Cite sources when possible  
- Ask clarifying questions if the request is unclear
- Adapt your communication style to match the user's needs
- Be honest about limitations or uncertainty

Focus on being helpful while maintaining accuracy and professionalism.`;

            editor = document.createElement('div');
            editor.id = 'custom-prompt-editor';
            editor.innerHTML = `
                <div style="margin: 1rem 0; padding: 1rem; border: 1px solid var(--pico-border-color); border-radius: var(--pico-border-radius); background: var(--pico-card-background-color);">
                    <label for="custom-prompt-input"><strong>Custom System Prompt:</strong></label>
                    <textarea id="custom-prompt-input" placeholder="Enter your custom system prompt..." rows="6" style="margin-top: 0.5rem;">${templatePrompt}</textarea>
                    <div style="margin-top: 0.5rem;">
                        <button type="button" id="save-custom-prompt" class="secondary">Save Custom Prompt</button>
                        <button type="button" id="clear-custom-prompt" class="outline">Clear & Reset Template</button>
                    </div>
                </div>
            `;
            chatWindow.parentNode.insertBefore(editor, chatWindow);
            
            // Add event listeners for the buttons
            document.getElementById('save-custom-prompt').addEventListener('click', () => {
                customSystemPrompt = document.getElementById('custom-prompt-input').value;
                addMessage('system', `Custom system prompt saved: "${customSystemPrompt || 'Empty'}"`);
            });
            
            document.getElementById('clear-custom-prompt').addEventListener('click', () => {
                const resetTemplate = `You are a knowledgeable assistant with expertise in multiple domains. Please:

- Provide accurate, well-researched information
- Cite sources when possible  
- Ask clarifying questions if the request is unclear
- Adapt your communication style to match the user's needs
- Be honest about limitations or uncertainty

Focus on being helpful while maintaining accuracy and professionalism.`;
                
                customSystemPrompt = '';
                document.getElementById('custom-prompt-input').value = resetTemplate;
                addMessage('system', 'Custom system prompt reset to template');
            });
        }
        editor.style.display = 'block';
    }

    function hideCustomPromptEditor() {
        const editor = document.getElementById('custom-prompt-editor');
        if (editor) {
            editor.style.display = 'none';
        }
    }

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
                    body: JSON.stringify({ 
                        prompt: prompt,
                        system_prompt: getCurrentSystemPrompt()
                    }),
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

        const endpoint = `${SERVER_URL}/generate/${selectedModel}`;

        // 1. Display the user's message immediately
        addMessage('user', prompt);

        // 2. Clear the input field and show a loading indicator
        promptInput.value = '';
        addMessage('model', 'Thinking...', true);

        // 3. Send the prompt to the backend API with retry logic
        await sendRequestWithRetry(endpoint, prompt);
    });
    
    // --- Helper function to get current system prompt ---
    function getCurrentSystemPrompt() {
        if (selectedSystemPrompt === 'custom') {
            return customSystemPrompt;
        } else {
            return systemPrompts[selectedSystemPrompt].prompt;
        }
    }
    
    // Initialize the interface
    populateSystemPrompts();
});
