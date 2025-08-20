<script lang="ts">
  import { onMount } from 'svelte';

  let mode: 'agents' | 'direct' = 'agents';
  let selectedModel: 'orchestrator' | 'medical' | 'clinical' = 'orchestrator';
  let prompt = '';
  let isSending = false;
  // Base URL for backend API. In dev, proxy handles it; in preview/build, set VITE_API_BASE_URL
  const SERVER_URL: string = (import.meta as any).env?.VITE_API_BASE_URL || '';
  console.log('SERVER_URL configured as:', SERVER_URL);

  import { marked } from 'marked';
  type ChatMsg = { sender: 'user' | 'model' | 'system'; text: string; html?: string };
  let messages: ChatMsg[] = [
    { sender: 'system', text: '👋 Welcome! A2A is the default. Ask a question to get started.' }
  ];

  // System prompts for Direct mode
  type SysPromptKey = 'default' | 'helpful' | 'concise' | 'medical' | 'researcher' | 'custom';
  let selectedPrompt: SysPromptKey = 'default';
  let customPrompt = '';
  const prompts: Record<SysPromptKey, { name: string; text: string }> = {
    default:   { name: '💬 Default',   text: '' },
    helpful:   { name: '🤝 Helpful',   text: 'You are a helpful, harmless, and honest assistant.' },
    concise:   { name: '⚡ Concise',   text: 'Be brief and direct in your responses.' },
    medical:   { name: '🏥 Medical',   text: 'You are a medical assistant. Provide educational information only - not medical advice.' },
    researcher:{ name: '🔬 Researcher',text: 'Provide evidence-based, well-researched responses.' },
    custom:    { name: '✏️ Custom',    text: '' }
  };
  function currentSystemPrompt(): string { return selectedPrompt === 'custom' ? customPrompt : prompts[selectedPrompt].text; }
  function selectPrompt(k: string) {
    selectedPrompt = (k as SysPromptKey);
  }

  async function callApi() {
    if (!prompt.trim()) return;
    isSending = true;
    messages = [...messages, { sender: 'user', text: prompt }];
    const thisPrompt = prompt;
    prompt = '';
    try {
      let res: Response;
      if (mode === 'agents') {
        res = await fetch(`${SERVER_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: thisPrompt, conversation_id: `conv_${Date.now()}` })
        });
      } else {
        res = await fetch(`${SERVER_URL}/generate/${selectedModel}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: thisPrompt, system_prompt: currentSystemPrompt(), conversation_history: [] })
        });
      }
      const data = await res.json();
      const text = data.response || JSON.stringify(data);
      const html = marked.parse(text ?? '');
      messages = [...messages, { sender: 'model', text, html }];
    } catch (e: any) {
      const err = `Error: ${e?.message || e}`;
      messages = [...messages, { sender: 'model', text: err, html: marked.parse(err) }];
    } finally {
      isSending = false;
      setTimeout(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }), 50);
    }
  }

  onMount(() => {
    mode = 'agents';
  });

  function setModeAgents() { mode = 'agents'; }
  function setModeDirect() { mode = 'direct'; }
  function setModelOrchestrator() { selectedModel = 'orchestrator'; }
  function setModelMedical() { selectedModel = 'medical'; }
  function setModelClinical() { selectedModel = 'clinical'; }
  function clearHistory() {
    messages = [ { sender: 'system', text: 'Conversation history cleared. Starting fresh conversation.' } ];
  }
</script>

<main class="container">
  <header>
    <nav>
      <ul>
        <li><h1>🤖 Multi-Model Chat</h1></li>
      </ul>
      <ul>
        <li>
          <details class="dropdown">
            <summary><span>{mode === 'agents' ? '🕸️ Agents (A2A)' : '🧠 Direct'}</span></summary>
            <ul>
              <li><button class="link-like" on:click|preventDefault={setModeDirect}>🧠 Direct (per-model)</button></li>
              <li><button class="link-like" on:click|preventDefault={setModeAgents}>🕸️ Agents (A2A)</button></li>
            </ul>
          </details>
        </li>
        {#if mode === 'direct'}
        <li>
          <details class="dropdown">
            <summary><span>{
              selectedModel === 'orchestrator' ? '🎯 Llama 3.1 8B' :
              selectedModel === 'medical' ? '🏥 MedGemma 4B' :
              '🔬 Gemma 3 1B'
            }</span></summary>
            <ul>
              <li><button class="link-like" on:click|preventDefault={setModelOrchestrator}>🎯 Llama 3.1 8B (Orchestrator)</button></li>
              <li><button class="link-like" on:click|preventDefault={setModelMedical}>🏥 MedGemma 4B (Medical)</button></li>
              <li><button class="link-like" on:click|preventDefault={setModelClinical}>🔬 Gemma 3 1B (Clinical Research)</button></li>
            </ul>
          </details>
        </li>
        <li>
          <details class="dropdown">
            <summary><span>{prompts[selectedPrompt].name}</span></summary>
            <ul>
              {#each Object.entries(prompts) as [key, p]}
                <li><button class="link-like" on:click|preventDefault={() => selectPrompt(key)}>{p.name}</button></li>
              {/each}
            </ul>
          </details>
        </li>
        {/if}
      </ul>
    </nav>
  </header>

  <section class="chat-window" id="chat-window">
    {#each messages as m}
      {#if m.sender === 'user'}
        <article class="message user-message"><div class="message-content">{m.text}</div></article>
      {:else if m.sender === 'model'}
        <article class="message model-message">
          <div class="message-content">{@html m.html ?? m.text}</div>
        </article>
      {:else}
        <article class="message-welcome"><p>{m.text}</p></article>
      {/if}
    {/each}
  </section>

  <form class="chat-form" on:submit|preventDefault={callApi}>
    <div class="form-content">
      <fieldset role="group">
        <input type="text" id="prompt-input" placeholder="Type your message here..." autocomplete="off" required bind:value={prompt} />
        <div class="button-group">
            <button type="submit" disabled={isSending}>📤 Send</button>
            <button type="button" id="clear-history" class="secondary" on:click={clearHistory}>🗑️ Clear</button>
        </div>
      </fieldset>
    </div>
  </form>
  <footer class="bottom-footer"><small>Powered by Svelte + Vite</small></footer>
</main>

<style>
  /* Component has no local styles; everything is in src/styles.css */
</style>


