document.addEventListener('DOMContentLoaded', () => {
    // Configuration - can be easily modified for different environments
    const SERVER_URL = 'http://127.0.0.1:3000';
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // milliseconds
    const REQUEST_TIMEOUT = 120000; // 2 minutes timeout for model generation
    
    // Get references to all the HTML elements we'll need to interact with
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const chatWindow = document.getElementById('chat-window');
    const currentModelSpan = document.getElementById('current-model');
    const currentPromptSpan = document.getElementById('current-prompt');
    const systemPromptsContainer = document.getElementById('system-prompts');
    const clearHistoryButton = document.getElementById('clear-history');
    const tabModeSpan = document.getElementById('tab-mode');
    const directControls = document.getElementById('direct-controls');
    const promptControls = document.getElementById('prompt-controls');
    
    // Current selections
    let selectedModel = 'general';
    let selectedSystemPrompt = 'default';
    let customSystemPrompt = '';
    let mode = 'direct'; // 'direct' | 'agents'
    
    // Conversation history management
    let conversationHistory = [];
    let conversationId = generateConversationId();
    
    // Request state management
    let isRequestInProgress = false;

    // System prompt presets
    const systemPrompts = {
        'default': { name: '💬 Default', prompt: '' },
        'helpful': { name: '🤝 Helpful', prompt: 'You are a helpful, harmless, and honest assistant.' },
        'concise': { name: '⚡ Concise', prompt: 'Be brief and direct in your responses.' },
        'medical': { name: '🏥 Medical', prompt: 'You are a medical assistant. Provide educational information only - not medical advice.' },
        'researcher': { name: '🔬 Researcher', prompt: 'Provide evidence-based, well-researched responses.' },
        'custom': { name: '✏️ Custom', prompt: '' }
    };

    // --- Conversation history management functions ---
    function generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function addToConversationHistory(role, content) {
        const lastMessage = conversationHistory[conversationHistory.length - 1];
        if (lastMessage && lastMessage.role === role && lastMessage.content === content) {
            console.log('Skipping duplicate message');
            return;
        }
        const message = {
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        };
        conversationHistory.push(message);
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }
        console.log(`Added ${role} message to history. Total messages: ${conversationHistory.length}`);
    }

    function clearConversationHistory() {
        conversationHistory = [];
        conversationId = generateConversationId();
        console.log('Conversation history cleared, new conversation started');
        addMessage('system', 'Conversation history cleared. Starting fresh conversation.');
    }

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
            if (selectedSystemPrompt === 'custom') {
                showCustomPromptEditor();
            } else {
                hideCustomPromptEditor();
            }
            const dropdown = e.target.closest('details');
            if (dropdown) dropdown.open = false;
        }

        if (e.target.matches('[data-mode]')) {
            e.preventDefault();
            mode = e.target.dataset.mode; // 'direct' or 'agents'
            tabModeSpan.textContent = mode === 'direct' ? '🧠 Direct' : '🕸️ Agents (A2A)';
            // Toggle visibility of controls relevant to direct mode
            directControls.style.display = mode === 'direct' ? 'block' : 'none';
            promptControls.style.display = mode === 'direct' ? 'block' : 'none';
            const dropdown = e.target.closest('details');
            if (dropdown) dropdown.open = false;
            addMessage('system', mode === 'direct' ? 'Switched to Direct mode.' : 'Switched to Agents (A2A) mode. Routing will be handled by agents.');
        }
    });

    // --- Custom prompt editor functions ---
    function showCustomPromptEditor() {
        let editor = document.getElementById('custom-prompt-editor');
        if (!editor) {
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
            document.getElementById('save-custom-prompt').addEventListener('click', () => {
                customSystemPrompt = document.getElementById('custom-prompt-input').value;
                addMessage('system', `Custom system prompt saved: "${customSystemPrompt || 'Empty'}"`);
                hideCustomPromptEditor();
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
        if (sender === 'model' && typeof marked !== 'undefined') {
            try {
                const renderedMarkdown = marked.parse(messageContent);
                contentDiv.innerHTML = renderedMarkdown;
            } catch (error) {
                console.warn('Markdown rendering failed, falling back to plain text:', error);
                const p = document.createElement('p');
                p.textContent = messageContent;
                contentDiv.appendChild(p);
            }
        } else {
            const p = document.createElement('p');
            p.textContent = messageContent;
            contentDiv.appendChild(p);
        }
        if (isLoading) {
            contentDiv.classList.add('loading-indicator');
            messageDiv.id = 'loading-message';
        }
        messageDiv.appendChild(contentDiv);
        chatWindow.appendChild(messageDiv);
        setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }, 100);
    }

    function removeLoadingMessage() {
        const loadingMessage = document.getElementById('loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    function fetchWithTimeout(url, options, timeout = REQUEST_TIMEOUT) {
        return Promise.race([
            fetch(url, options),
            new Promise((_, reject) => setTimeout(() => reject(new Error('Request timeout - model may be processing')), timeout))
        ]);
    }

    async function checkServerHealth() {
        try {
            const response = await fetch(`${SERVER_URL}/health`, { method: 'GET', timeout: 5000 });
            if (response.ok) {
                const data = await response.json();
                console.log('Server health:', data);
                return data;
            }
        } catch (error) {
            console.error('Server health check failed:', error);
            return null;
        }
    }

    async function sendRequestWithRetry(endpoint, bodyObj) {
        let lastError = null;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                console.log(`Sending request to ${endpoint} (attempt ${attempt}/${MAX_RETRIES})`);
                const startTime = Date.now();
                const response = await fetchWithTimeout(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bodyObj),
                });
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || response.statusText);
                }
                const data = await response.json();
                const executionTime = Date.now() - startTime;
                console.log(`Request completed in ${executionTime}ms`);
                removeLoadingMessage();
                addMessage('model', data.response);
                addToConversationHistory('user', bodyObj.prompt);
                addToConversationHistory('assistant', data.response);
                return;
            } catch (error) {
                lastError = error;
                console.error(`Attempt ${attempt} failed:`, error);
                if (attempt < MAX_RETRIES) {
                    await sleep(RETRY_DELAY * attempt);
                }
            }
        }
        removeLoadingMessage();
        addMessage('model', `Error after ${MAX_RETRIES} attempts: ${lastError.message}`);
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isRequestInProgress) {
            console.log('Request already in progress, ignoring submission');
            return;
        }
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        isRequestInProgress = true;
        const submitButton = chatForm.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.textContent = '⏳ Sending...';
        promptInput.disabled = true;
        clearHistoryButton.disabled = true;

        try {
            addMessage('user', prompt);
            const maxHistoryBeforeSummary = mode === 'agents' ? 8 : (selectedModel === 'medgemma' ? 16 : 16);
            if (conversationHistory.length > maxHistoryBeforeSummary && conversationHistory.length % 10 === 0) {
                addMessage('system', `Context management notice: conversation length is ${conversationHistory.length}.`);
            }
            promptInput.value = '';
            addMessage('model', 'Thinking...', true);

            if (mode === 'agents') {
                await sendRequestWithRetry(`${SERVER_URL}/chat`, {
                    prompt: prompt,
                    conversation_id: conversationId,
                });
            } else {
                const endpoint = `${SERVER_URL}/generate/${selectedModel}`;
                await sendRequestWithRetry(endpoint, {
                    prompt: prompt,
                    system_prompt: getCurrentSystemPrompt(),
                    conversation_history: conversationHistory,
                    conversation_id: conversationId,
                });
            }
        } finally {
            isRequestInProgress = false;
            submitButton.disabled = false;
            submitButton.textContent = originalButtonText;
            promptInput.disabled = false;
            clearHistoryButton.disabled = false;
            promptInput.focus();
        }
    });
    
    function getCurrentSystemPrompt() {
        if (selectedSystemPrompt === 'custom') return customSystemPrompt;
        return systemPrompts[selectedSystemPrompt].prompt;
    }
    
    clearHistoryButton.addEventListener('click', () => {
        if (isRequestInProgress) {
            addMessage('system', 'Cannot clear history while a request is in progress.');
            return;
        }
        if (conversationHistory.length === 0) {
            addMessage('system', 'No conversation history to clear.');
            return;
        }
        if (confirm('Clear conversation history?')) {
            clearConversationHistory();
        }
    });

    populateSystemPrompts();
    checkServerHealth().then((health) => {
        if (health) {
            console.log('Server healthy.');
        } else {
            console.warn('Server health check failed - may be starting up');
        }
    });
});
