# Voice-Interactive Blackjack Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace numpad/text card input with always-on ElevenLabs Scribe v2 streaming speech-to-text and browser SpeechSynthesis voice advisories.

**Architecture:** Single HTML file (`bj/index.html`). Audio captured via AudioWorklet, streamed over WebSocket to ElevenLabs Scribe v2 for real-time transcription. Committed transcripts feed the existing `parseDictation()` → `tapCard()`/`shuffle()`/`undo()` pipeline. Browser `speechSynthesis` speaks bet tier changes. No new files — all changes within the UI layer.

**Tech Stack:** Web Audio API (AudioWorkletNode), ElevenLabs Scribe v2 WebSocket API, Web Speech API (SpeechSynthesis), vanilla JS/HTML/CSS.

**Spec:** `docs/superpowers/specs/2026-05-09-voice-blackjack-companion-design.md`

---

## File Map

All changes are in **`bj/index.html`** (single file). Changes organized by section:

| Section | Lines (approx) | What Changes |
|---------|----------------|--------------|
| CSS: numpad styles | 141–183 | Remove entirely |
| CSS: keymap-card styles | 234–236 | Remove entirely |
| CSS: actions-row | 139 | Update to 4-column |
| CSS: new mic styles | (insert after 138) | Add mic indicator, pulse animation |
| HTML: setup wizard | 259 (after ramp input) | Add API key field |
| HTML: voice-card | 292–301 | Replace with mic indicator + transcript log |
| HTML: numpad card | 321–354 | Remove entirely |
| HTML: keymap card | 356–359 | Remove entirely |
| HTML: actions-row | 349–354 | Add mic toggle button |
| JS: setup functions | 1231–1281 | Add API key read/write/default |
| JS: voice/dictation | 1425–1456 | Replace with Scribe integration |
| JS: keyboard shortcuts | 1458–1489 | Remove numpad/rank key handlers, keep undo (U) and shuffle (X) |
| JS: new modules | (insert before boot) | AudioCapture, ScribeWS, TTS modules |

---

### Task 1: Remove Numpad CSS and HTML

**Files:**
- Modify: `bj/index.html:141-183` (CSS), `bj/index.html:234-236` (CSS), `bj/index.html:321-359` (HTML)

- [ ] **Step 1: Remove numpad CSS (lines 141–183)**

Delete the entire block from `/* Numpad-style keypad */` through `.numpad-title { ... }`. This is 43 lines of grid-area assignments, color coding, and button sizing for the numpad that no longer exists.

- [ ] **Step 2: Remove keymap-card CSS (lines 234–236)**

Delete the three `.keymap-card` rules.

- [ ] **Step 3: Remove numpad HTML (lines 321–354)**

Delete from `<!-- Numpad-style keypad -->` through the closing `</div>` of that card, including the actions-row inside it. The actions-row will be recreated in a later task with the mic toggle.

- [ ] **Step 4: Remove keymap HTML (lines 356–359)**

Delete the `<div class="card keymap-card">` block.

- [ ] **Step 5: Remove voice-card HTML (lines 292–301)**

Delete the `<div class="card voice-card">` block (the Wispr Flow text input + Apply button + voice log div).

- [ ] **Step 6: Run smoke test to verify the page still loads**

Run: `node bj/smoke.js`
Expected: No parse errors. The app should load, show setup or game screen.

- [ ] **Step 7: Commit**

```bash
git add bj/index.html
git commit -m "Remove numpad, keymap, and text voice input UI elements"
```

---

### Task 2: Remove Numpad JS Event Handlers

**Files:**
- Modify: `bj/index.html:1406-1489` (JS event binding section)

- [ ] **Step 1: Remove numpad button event bindings (lines 1412–1422)**

Delete these lines that wire up `data-rank`, `data-rc`/`data-cards`, and `data-action` button clicks:
```javascript
$$('button[data-rank]').forEach(b => b.onclick = () => tapCard(b.dataset.rank));
$$('button[data-rc][data-cards]').forEach(b => {
  b.onclick = () => quickAdjust(parseInt(b.dataset.rc, 10), parseInt(b.dataset.cards, 10));
});
$$('button[data-action]').forEach(b => {
  b.onclick = () => {
    const a = b.dataset.action;
    if (a === 'undo') undo();
    else if (a === 'shuffle') shuffle();
  };
});
```

- [ ] **Step 2: Remove voice text input handlers (lines 1448–1456)**

Delete the `$('#voice-send').onclick` handler and the `$('#voice-input')` keydown listener. Keep the `applyDictation()` function (lines 1425–1447) — it's still used by the new Scribe integration.

- [ ] **Step 3: Trim keyboard shortcuts to only Undo and Shuffle (lines 1458–1489)**

Replace the entire `document.addEventListener('keydown', ...)` handler. Remove all card rank key handlers (digits 0–9, letters a/t), all numpad operator handlers (+, -, *, /, ., Enter, Clear), and all laptop backup handlers (=, ], [, comma). Keep only:

```javascript
document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if ($('#game-screen').classList.contains('hide')) return;
  const k = e.key;
  if (k === 'u' || k === 'U' || k === 'Backspace') { e.preventDefault(); undo(); }
  if (k === 'x' || k === 'X') { e.preventDefault(); shuffle(); }
});
```

- [ ] **Step 4: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 5: Run existing unit tests to verify Lib is untouched**

Run: `node bj/test-runner.js`
Expected: All 118+ tests pass. Zero failures.

- [ ] **Step 6: Commit**

```bash
git add bj/index.html
git commit -m "Remove numpad and text input JS handlers, keep undo/shuffle keys"
```

---

### Task 3: Add Mic Indicator UI and CSS

**Files:**
- Modify: `bj/index.html` — CSS section (insert new styles), HTML section (add mic indicator + actions row)

- [ ] **Step 1: Update `.actions-row` CSS (line 139)**

Change from:
```css
.actions-row { display: grid; grid-template-columns: 1fr 1fr auto; gap: 6px; }
```
To:
```css
.actions-row { display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 6px; }
```

- [ ] **Step 2: Add mic indicator CSS**

Insert after the `.actions-row` rule (around line 140). Add these styles:

```css
/* Mic status indicator */
.mic-card { text-align: center; }
.mic-status { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 14px; }
.mic-dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--bad); flex-shrink: 0;
}
.mic-dot.on { background: var(--good); animation: mic-pulse 1.5s ease-in-out infinite; }
.mic-dot.connecting { background: var(--warn); animation: mic-pulse 0.8s ease-in-out infinite; }
@keyframes mic-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.5; transform: scale(1.3); }
}
.mic-label { font-size: 14px; font-weight: 600; }
.mic-partial { font-size: 13px; color: var(--muted); font-style: italic; min-height: 1.4em; padding: 0 12px; }
.mic-log { font-size: 12px; color: var(--muted); min-height: 1.2em; padding: 4px 12px 8px; }
```

- [ ] **Step 3: Add mic indicator HTML**

Insert in the game screen, after the active deviations card (after line 319 — the `</div>` closing `devs-card`), and before where the numpad used to be. Add:

```html
<!-- Mic status + transcript -->
<div class="card mic-card">
  <div class="mic-status">
    <div class="mic-dot" id="mic-dot"></div>
    <span class="mic-label" id="mic-label">Mic off</span>
  </div>
  <div class="mic-partial" id="mic-partial"></div>
  <div class="mic-log" id="mic-log">Tap mic or press M to start listening.</div>
</div>
```

- [ ] **Step 4: Add actions row HTML with mic toggle**

Insert after the mic card:

```html
<div class="card" style="background:transparent;border:none;padding:0">
  <div class="actions-row">
    <button id="btn-undo" class="ghost">↶ Undo <span class="kbd">U</span></button>
    <button id="btn-shuffle" class="warn">Shuffle <span class="kbd">X</span></button>
    <button id="btn-mic" class="primary">🎤 Mic <span class="kbd">M</span></button>
    <button id="btn-settings" class="ghost">⚙</button>
  </div>
</div>
```

- [ ] **Step 5: Add M key shortcut for mic toggle**

In the keyboard shortcut handler (from Task 2), add:

```javascript
if (k === 'm' || k === 'M') { e.preventDefault(); toggleMic(); }
```

Add a placeholder `toggleMic()` function near the top of the UI section:

```javascript
function toggleMic() { /* wired in Task 5 */ }
```

- [ ] **Step 6: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 7: Commit**

```bash
git add bj/index.html
git commit -m "Add mic status indicator, transcript log, and actions row with mic toggle"
```

---

### Task 4: Add API Key to Setup Wizard

**Files:**
- Modify: `bj/index.html` — HTML setup section, JS `showSetup()`, `readSetup()`, `defaults()`

- [ ] **Step 1: Add API key HTML field in setup wizard**

Insert after the Betting card's closing `</div>` (after line 268, before the button row), add:

```html
<div class="card">
  <h2>Voice</h2>
  <label>ElevenLabs API Key
    <input type="password" id="cfg-apikey" placeholder="xi-...">
  </label>
  <small style="color:var(--muted)">Required for voice input. Stored locally, never sent to any server except ElevenLabs.</small>
</div>
```

- [ ] **Step 2: Update `showSetup()` to prefill API key**

In `showSetup()` (around line 1231), add after `$('#cfg-ramp').value = c.ramp.join(',');`:

```javascript
if (c.elevenLabsKey) $('#cfg-apikey').value = c.elevenLabsKey;
```

- [ ] **Step 3: Update `readSetup()` to include API key**

In `readSetup()` (around line 1245), add to the returned object:

```javascript
elevenLabsKey: $('#cfg-apikey').value.trim(),
```

- [ ] **Step 4: Update `defaults()` to clear API key**

In `defaults()` (around line 1273), add:

```javascript
$('#cfg-apikey').value = '';
```

- [ ] **Step 5: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 6: Run unit tests**

Run: `node bj/test-runner.js`
Expected: All tests pass (Lib unchanged).

- [ ] **Step 7: Commit**

```bash
git add bj/index.html
git commit -m "Add ElevenLabs API key field to setup wizard"
```

---

### Task 5: Implement Audio Capture Module

**Files:**
- Modify: `bj/index.html` — JS section (insert AudioCapture module before UI boot)

- [ ] **Step 1: Write the AudioCapture module**

Insert **inside** the `UI` IIFE (after `const L = Lib;` around line 1216), not between `State` and the UI IIFE. This is critical — placing browser-only modules (AudioCapture, ScribeWS, TTS) outside the UI IIFE would break the Node-based test runner and smoke tests, which truncate code at the `if (typeof document !== 'undefined')` guard and run the rest in a VM context without browser globals like `AudioContext`, `WebSocket`, `speechSynthesis`, etc.

All three modules (AudioCapture, ScribeWS, TTS) must be inside the UI IIFE.

```javascript
// ============================================================================
// AUDIO — mic capture → 16 kHz mono PCM → base64 chunks
// ============================================================================
const AudioCapture = (() => {
  let ctx, stream, worklet;
  let onChunk = null; // callback: (base64String) => void

  const PROCESSOR_CODE = `
    class PCMProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this._buf = [];
        this._count = 0;
      }
      process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const samples = input[0]; // Float32Array, native sample rate
        // Accumulate ~100ms of audio before posting
        for (let i = 0; i < samples.length; i++) {
          this._buf.push(samples[i]);
        }
        this._count += samples.length;
        if (this._count >= sampleRate / 10) { // ~100ms
          this.port.postMessage({ samples: new Float32Array(this._buf) });
          this._buf = [];
          this._count = 0;
        }
        return true;
      }
    }
    registerProcessor('pcm-processor', PCMProcessor);
  `;

  function resample(float32, fromRate, toRate) {
    if (fromRate === toRate) return float32;
    const ratio = fromRate / toRate;
    const newLen = Math.round(float32.length / ratio);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, float32.length - 1);
      const frac = srcIdx - lo;
      out[i] = float32[lo] * (1 - frac) + float32[hi] * frac;
    }
    return out;
  }

  function floatToPCM16(float32) {
    const buf = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buf;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  async function start(chunkCallback) {
    onChunk = chunkCallback;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true }
    });
    ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    const blob = new Blob([PROCESSOR_CODE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    worklet = new AudioWorkletNode(ctx, 'pcm-processor');
    worklet.port.onmessage = (e) => {
      const resampled = resample(e.data.samples, ctx.sampleRate, 16000);
      const pcm = floatToPCM16(resampled);
      const b64 = arrayBufferToBase64(pcm);
      if (onChunk) onChunk(b64);
    };
    source.connect(worklet);
    worklet.connect(ctx.destination); // required for processing, outputs silence
  }

  function stop() {
    if (worklet) { worklet.disconnect(); worklet = null; }
    if (ctx) { ctx.close(); ctx = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    onChunk = null;
  }

  function isActive() { return !!stream; }

  return { start, stop, isActive };
})();
```

- [ ] **Step 2: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors. (AudioCapture uses browser-only APIs, but the module is guarded by not auto-starting.)

- [ ] **Step 3: Commit**

```bash
git add bj/index.html
git commit -m "Add AudioCapture module: mic → 16kHz PCM → base64 chunks"
```

---

### Task 6: Implement Scribe WebSocket Module

**Files:**
- Modify: `bj/index.html` — JS section (insert ScribeWS module after AudioCapture)

- [ ] **Step 1: Write the ScribeWS module**

Insert after the AudioCapture module. This module handles token fetch, WebSocket lifecycle, and transcript events:

```javascript
// ============================================================================
// SCRIBE — ElevenLabs Scribe v2 realtime WebSocket
// ============================================================================
const ScribeWS = (() => {
  let ws = null;
  let retries = 0;
  const MAX_RETRIES = 3;
  const BASE_DELAY = 2000;

  // Callbacks set by consumer
  let onSessionStarted = null;
  let onPartial = null;
  let onCommit = null;
  let onError = null;
  let onClose = null;

  async function fetchToken(apiKey) {
    const resp = await fetch('https://api.elevenlabs.io/v1/single-use-token/realtime_scribe', {
      method: 'POST',
      headers: { 'xi-api-key': apiKey },
    });
    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error(`Token fetch failed (${resp.status}): ${body}`);
    }
    const data = await resp.json();
    return data.token;
  }

  function buildURL(token) {
    const params = new URLSearchParams({
      token,
      model_id: 'scribe_v2_realtime',
      audio_format: 'pcm_16000',
      commit_strategy: 'vad',
      vad_silence_threshold_secs: '0.8',
      vad_threshold: '0.4',
      language_code: 'en',
      keyterms: 'ace,jack,queen,king,shuffle,undo',
    });
    return `wss://api.elevenlabs.io/v1/speech-to-text/realtime?${params}`;
  }

  async function connect(apiKey) {
    close();
    const token = await fetchToken(apiKey);
    const url = buildURL(token);
    ws = new WebSocket(url);

    ws.onopen = () => { retries = 0; };

    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      switch (msg.message_type) {
        case 'session_started':
          if (onSessionStarted) onSessionStarted(msg);
          break;
        case 'partial_transcript':
          if (onPartial) onPartial(msg.text);
          break;
        case 'committed_transcript':
          if (onCommit) onCommit(msg.text);
          break;
        case 'auth_error':
        case 'quota_exceeded':
        case 'rate_limited':
        case 'input_error':
        case 'transcriber_error':
        case 'error':
          if (onError) onError(msg.message_type, msg.error || 'Unknown error');
          break;
        case 'session_time_limit_exceeded':
          // Auto-reconnect seamlessly
          if (onError) onError('session_time_limit_exceeded', 'Reconnecting...');
          reconnect(apiKey);
          break;
      }
    };

    ws.onerror = () => {
      if (onError) onError('network', 'WebSocket error');
    };

    ws.onclose = (e) => {
      if (onClose) onClose(e.code, e.reason);
    };
  }

  async function reconnect(apiKey) {
    while (retries < MAX_RETRIES) {
      retries++;
      const delay = Math.min(BASE_DELAY * Math.pow(2, retries - 1), 8000);
      await new Promise(r => setTimeout(r, delay));
      try {
        await connect(apiKey);
        return; // success — onopen will reset retries
      } catch (err) {
        if (onError) onError('reconnect_failed', err.message);
      }
    }
    if (onError) onError('max_retries', 'Connection lost. Tap mic to reconnect.');
  }

  function sendAudio(base64) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        message_type: 'input_audio_chunk',
        audio_base_64: base64,
      }));
    }
  }

  function close() {
    if (ws) {
      ws.onclose = null; // prevent reconnect trigger
      ws.close();
      ws = null;
    }
  }

  function isConnected() { return ws && ws.readyState === WebSocket.OPEN; }
  function resetRetries() { retries = 0; }

  return {
    connect, close, sendAudio, isConnected, resetRetries, reconnect,
    set onSessionStarted(fn) { onSessionStarted = fn; },
    set onPartial(fn) { onPartial = fn; },
    set onCommit(fn) { onCommit = fn; },
    set onError(fn) { onError = fn; },
    set onClose(fn) { onClose = fn; },
  };
})();
```

- [ ] **Step 2: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 3: Commit**

```bash
git add bj/index.html
git commit -m "Add ScribeWS module: token fetch, WebSocket lifecycle, transcript events"
```

---

### Task 7: Implement TTS Advisory Module

**Files:**
- Modify: `bj/index.html` — JS section (insert TTS module after ScribeWS)

- [ ] **Step 1: Write the TTS module**

Insert after ScribeWS:

```javascript
// ============================================================================
// TTS — browser SpeechSynthesis for short advisories
// ============================================================================
const TTS = (() => {
  let voice = null;
  let lastTier = null;

  // Resolve preferred voice after browser loads voice list
  function init() {
    const pick = () => {
      const voices = speechSynthesis.getVoices();
      voice = voices.find(v => v.lang.startsWith('en') && v.localService) ||
              voices.find(v => v.lang.startsWith('en')) ||
              voices[0] || null;
    };
    pick();
    if (typeof speechSynthesis !== 'undefined' && speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = pick;
    }
  }

  function speak(text) {
    if (typeof speechSynthesis === 'undefined' || !text) return;
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1.3;
    if (voice) utt.voice = voice;
    speechSynthesis.cancel(); // cancel any queued speech
    speechSynthesis.speak(utt);
  }

  // Called after render() with current bet tier string.
  // Only speaks when tier changes.
  function adviseBetTier(tier) {
    if (tier === lastTier) return;
    lastTier = tier;
    const phrases = {
      cold: 'Min bet',
      flat: 'Min bet',
      warm: 'Ramp up',
      hot: 'Bet big',
      max: 'Bet max',
    };
    if (phrases[tier]) speak(phrases[tier]);
  }

  function adviseCutCard() {
    speak('Cut card');
  }

  function adviseInsurance(take) {
    speak(take ? 'Take insurance' : 'Skip insurance');
  }

  function reset() { lastTier = null; }

  return { init, speak, adviseBetTier, adviseCutCard, adviseInsurance, reset };
})();
```

- [ ] **Step 2: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 3: Commit**

```bash
git add bj/index.html
git commit -m "Add TTS module: bet tier, cut card, and insurance advisories"
```

---

### Task 8: Wire Everything Together in the UI

**Files:**
- Modify: `bj/index.html` — JS UI section (replace old voice handlers, wire AudioCapture + ScribeWS + TTS)

This is the integration task. It connects the three new modules to the existing UI lifecycle.

- [ ] **Step 1: Add mic state tracking and `toggleMic()` implementation**

Replace the placeholder `toggleMic()` with the full implementation. Add near the top of the UI IIFE (after `const L = Lib;`):

```javascript
let micActive = false;
let micStarting = false; // guard against double-click during async startup
let currentApiKey = null;

async function toggleMic() {
  if (micStarting) return; // prevent double-click
  if (micActive) {
    stopMic();
  } else {
    await startMic();
  }
}

async function startMic() {
  const s = State.get();
  if (!s.setup || !s.setup.elevenLabsKey) {
    updateMicUI('error', 'No API key — check settings');
    return;
  }
  micStarting = true;
  currentApiKey = s.setup.elevenLabsKey;
  updateMicUI('connecting', 'Connecting...');

  try {
    // Wire ScribeWS callbacks
    ScribeWS.onSessionStarted = () => {
      micActive = true;
      micStarting = false;
      updateMicUI('on', 'Listening...');
    };
    ScribeWS.onPartial = (text) => {
      $('#mic-partial').textContent = text;
    };
    ScribeWS.onCommit = (text) => {
      $('#mic-partial').textContent = '';
      applyDictation(text);
      checkAdvisories();
    };
    ScribeWS.onError = (type, msg) => {
      if (type === 'auth_error') {
        stopMic();
        updateMicUI('error', 'Invalid API key');
      } else if (type === 'quota_exceeded') {
        stopMic();
        updateMicUI('error', 'API quota exceeded');
      } else if (type === 'max_retries') {
        stopMic();
        updateMicUI('error', msg);
      } else if (type === 'session_time_limit_exceeded') {
        updateMicUI('connecting', 'Reconnecting...');
      } else if (type === 'rate_limited') {
        updateMicUI('connecting', 'Rate limited — retrying...');
        setTimeout(() => ScribeWS.reconnect(currentApiKey), 5000);
      }
    };
    ScribeWS.onClose = () => {
      if (micActive) {
        updateMicUI('connecting', 'Reconnecting...');
        ScribeWS.reconnect(currentApiKey);
      }
    };

    // Start audio capture → ScribeWS
    await AudioCapture.start((b64) => ScribeWS.sendAudio(b64));
    await ScribeWS.connect(currentApiKey);
  } catch (err) {
    micStarting = false;
    stopMic();
    const msg = (err.name === 'NotAllowedError') ? 'Mic access required — check browser permissions' : (err.message || 'Mic error');
    updateMicUI('error', msg);
  }
}

function stopMic() {
  micActive = false;
  micStarting = false;
  AudioCapture.stop();
  ScribeWS.close();
  ScribeWS.resetRetries();
  updateMicUI('off', 'Mic off');
  $('#mic-partial').textContent = '';
}

function updateMicUI(state, label) {
  const dot = $('#mic-dot');
  const lbl = $('#mic-label');
  dot.className = 'mic-dot';
  if (state === 'on') dot.classList.add('on');
  else if (state === 'connecting') dot.classList.add('connecting');
  lbl.textContent = label;
  // Update mic button appearance
  const btn = $('#btn-mic');
  if (btn) {
    btn.classList.toggle('primary', !micActive);
    btn.classList.toggle('bad', micActive);
    btn.innerHTML = micActive
      ? '⏹ Stop <span class="kbd">M</span>'
      : '🎤 Mic <span class="kbd">M</span>';
  }
}
```

- [ ] **Step 2: Add `checkAdvisories()` function**

This function checks for state changes that should trigger TTS. Add after `updateMicUI`:

```javascript
let prevCutCardReached = false;

function checkAdvisories() {
  const s = State.get();
  if (!s.setup) return;
  const dr = Lib.decksRemaining(s.shoe.cardsDealt, s.shoe.totalDecks);
  const tc = Lib.trueCount(s.shoe.runningCount, dr);
  const units = Lib.wongUnits(tc, s.setup.ramp);
  const maxUnits = s.setup.ramp[s.setup.ramp.length - 1];

  // Bet tier
  let tier;
  if (tc < 0) tier = 'cold';
  else if (tc < 2) tier = 'flat';
  else if (units >= maxUnits) tier = 'max';
  else if (tc < 4) tier = 'warm';
  else tier = 'hot';
  TTS.adviseBetTier(tier);

  // Cut card
  const cutReached = s.shoe.cardsDealt >= s.shoe.shuffleAt;
  if (cutReached && !prevCutCardReached) TTS.adviseCutCard();
  prevCutCardReached = cutReached;
}
```

- [ ] **Step 3: Wire button click handlers**

In the event binding section, update the button wiring:

```javascript
$('#btn-undo').onclick    = undo;
$('#btn-shuffle').onclick = shuffle;
$('#btn-mic').onclick     = toggleMic;
$('#btn-settings').onclick = () => { stopMic(); State.patch(()=>{}); showSetup(State.get()); };
```

- [ ] **Step 4: Update `applyDictation()` to write to `#mic-log` instead of `#voice-log`**

In the existing `applyDictation()` function, change all references from `$('#voice-log')` to `$('#mic-log')`:

```javascript
// Line that was: $('#voice-log').textContent = 'Nothing heard.';
$('#mic-log').textContent = 'Nothing heard.';

// Line that was: $('#voice-log').textContent = msg;
$('#mic-log').textContent = msg;
```

- [ ] **Step 5: Initialize TTS on boot and reset on shuffle**

In the boot section (around line 1491), add `TTS.init();` after `State.load();`.

In the `shuffle()` function, add `TTS.reset();` after the state patch, so bet tier advisories re-trigger after a new shoe.

Also reset `prevCutCardReached = false;` in the `shuffle()` function.

- [ ] **Step 6: Run smoke test**

Run: `node bj/smoke.js`
Expected: No parse errors.

- [ ] **Step 7: Run unit tests**

Run: `node bj/test-runner.js`
Expected: All 118+ tests pass.

- [ ] **Step 8: Commit**

```bash
git add bj/index.html
git commit -m "Wire AudioCapture + ScribeWS + TTS into UI lifecycle"
```

---

### Task 9: Verify All Automated Tests Pass

**Files:**
- Possibly modify: `bj/integration.js`, `bj/smoke.js` (only if they fail due to DOM changes)

- [ ] **Step 1: Run unit tests**

Run: `node bj/test-runner.js`
Expected: All 118+ tests pass. These test `Lib` functions only — no DOM dependencies.

- [ ] **Step 2: Run integration tests**

Run: `node bj/integration.js`
Expected: All 18 scenarios pass. These tests call `Lib.parseDictation()` and count functions directly — no DOM element references. If any fail, investigate and fix the specific failure.

- [ ] **Step 3: Run smoke test**

Run: `node bj/smoke.js`
Expected: HTML parses without errors. Since AudioCapture, ScribeWS, and TTS are inside the UI IIFE (guarded by `typeof document !== 'undefined'`), they won't execute in Node.

- [ ] **Step 4: Run full suite together**

Run: `node bj/test-runner.js && node bj/integration.js && node bj/smoke.js`
Expected: All pass.

- [ ] **Step 5: Commit if any test files were modified**

```bash
git add bj/integration.js bj/smoke.js
git commit -m "Fix test compatibility with voice UI changes"
```
Skip this step if no test files needed changes.

---

### Task 10: Manual Browser Verification

**Files:** None (verification only)

- [ ] **Step 1: Open in browser and verify setup screen**

Open `bj/index.html` in Chrome. Verify:
- Setup screen shows all original fields (decks, penetration, H17, DAS, LS, unit, ramp)
- New "Voice" card with API key field appears
- "Start session" button works

- [ ] **Step 2: Verify game screen layout**

After starting a session (without API key), verify:
- Bet hero panel renders correctly
- Count strip shows RC/TC/Decks left
- Penetration bar renders
- Active deviations grid renders
- Mic indicator shows "Mic off" (red dot)
- Actions row has 4 buttons: Undo, Shuffle, Mic, Settings
- No numpad visible
- No text input visible
- No keymap card visible

- [ ] **Step 3: Verify mic connection (with API key)**

Enter a valid ElevenLabs API key in settings, restart session:
- Click Mic button → browser asks for mic permission
- After granting → dot turns green, "Listening..." appears
- Speak "ten six" → mic-log shows "Heard: 2 cards (RC 0)"
- RC/TC update on screen
- Click Mic again → stops, dot turns red

- [ ] **Step 4: Verify TTS advisories**

With mic active, rapidly feed high cards to push count up:
- When bet tier transitions (e.g., from flat to warm), browser should speak "Ramp up"
- When count goes high enough for max, should speak "Bet max"
- After shuffle (say "shuffle" or press X), tier resets

- [ ] **Step 5: Verify keyboard shortcuts**

- Press U → undo last card (works)
- Press X → shuffle (works)
- Press M → toggle mic
- Press digit keys → nothing happens (numpad shortcuts removed)

- [ ] **Step 6: Commit final verification note**

No code changes — just confirm all manual tests pass.
