# Voice-Interactive Blackjack Companion — Design Spec

**Date:** 2026-05-09
**Status:** Approved
**Scope:** Replace numpad/text input in BJ Companion with real-time voice interaction using ElevenLabs Scribe v2 streaming STT and browser SpeechSynthesis TTS.

---

## Problem

During live blackjack play, entering cards via numpad or typed dictation is too slow and requires attention away from the table. The user needs a hands-free, voice-driven interface: speak the cards as they appear, hear back actionable advisories, and glance at the screen for counts and deviations.

## Constraints

- Single HTML file (no build step, no server)
- Laptop browser (Chrome) — primary platform
- Personal tool — API key stored client-side in localStorage
- Existing `Lib` (pure counting/strategy functions) and `State` modules unchanged
- Existing `parseDictation()` function unchanged — already battle-tested for noisy input

---

## Architecture

Three new components replace the numpad and text input, all within the UI layer of `index.html`:

### 1. Audio Capture

- `navigator.mediaDevices.getUserMedia({ audio: true })` captures mic
- `AudioContext` with `AudioWorkletNode` downsamples to 16kHz mono PCM 16-bit (ScriptProcessorNode is deprecated and runs on the main thread — avoid it)
- AudioWorklet processor loaded via inline `Blob` URL to preserve the single-file constraint (no separate .js file)
- Each ~100ms chunk is base64-encoded and sent over WebSocket
- Mic captured with `{ audio: { echoCancellation: true } }` to prevent TTS output from being picked up as card input

### 2. ElevenLabs Scribe v2 WebSocket

**Endpoint:** `wss://api.elevenlabs.io/v1/speech-to-text/realtime`

**Connection parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model_id` | `scribe_v2_realtime` | Realtime-optimized Scribe model |
| `audio_format` | `pcm_16000` | 16kHz PCM matches capture pipeline |
| `commit_strategy` | `vad` | Auto-commit on silence (hands-free) |
| `vad_silence_threshold_secs` | `0.8` | Shorter than default 1.5s for snappy response |
| `vad_threshold` | `0.4` | Default voice activity sensitivity |
| `language_code` | `en` | English only — avoids language detection latency |
| `keyterms` | `["ace","jack","queen","king","shuffle","undo"]` | Boost recognition of card/command vocabulary |
**Authentication (two-step, no server needed):**

Browser `WebSocket` does not support custom headers, so `xi-api-key` cannot be sent directly. Instead:

1. **Fetch single-use token:** `fetch('https://api.elevenlabs.io/v1/single-use-token/realtime_scribe', { method: 'POST', headers: { 'xi-api-key': apiKey } })` — this works because `fetch()` supports custom headers. Returns a temporary token (valid 15 minutes, single use).
2. **Connect WebSocket with token:** `new WebSocket('wss://api.elevenlabs.io/v1/speech-to-text/realtime?token=<token>&model_id=scribe_v2_realtime&...')`

A fresh token must be fetched on every WebSocket connection or reconnection. The API key itself stays in localStorage and is only used for the token fetch — never exposed in a WebSocket URL.

**Client-to-server messages:**
```json
{
  "message_type": "input_audio_chunk",
  "audio_base_64": "<base64 PCM data>"
}
```

**Server-to-client messages used:**
- `session_started` — connection confirmed, start streaming
- `partial_transcript` — live preview of what Scribe is hearing (displayed to user)
- `committed_transcript` — finalized text after VAD silence detection (triggers action)
- Error messages — handled per type (see Error Handling)

### 3. Browser TTS (SpeechSynthesis)

Short spoken advisories using `speechSynthesis.speak()`. Rate set to 1.3x for urgency. Voice selected from `speechSynthesis.getVoices()` preferring an English voice; falls back to the default voice if none found. Handles the async `voiceschanged` event since `getVoices()` may return empty on first call.

**When it speaks:**
- Bet tier changes: "Min bet", "Ramp up", "Bet big", "Bet max"
- Insurance call when dealer shows ace: "Take insurance" / "Skip insurance" (TC >= 3)
- Shuffle alert: "Cut card" when penetration threshold reached

**When it stays silent:**
- Does not repeat back card names or count numbers (those are on screen)
- Does not speak on every transcript — only actionable state changes

---

## Transcript Processing Flow

1. `committed_transcript` received from Scribe
2. Text fed into `parseDictation()` (existing function, unchanged)
3. Each parsed action executed via existing functions:
   - `type: 'card'` → `tapCard(rank)`
   - `type: 'shuffle'` → `shuffle()`
   - `type: 'undo'` → `undo()`
   - `type: 'adjust'` → `quickAdjust(rc, cards)`
4. `render()` updates screen (counts, bet hero, deviations)
5. TTS speaks advisory if bet tier changed or actionable event occurred
6. Transcript log updated: "Heard: 4 cards (RC +2)"

**Undo granularity:** The existing `applyDictation()` calls `tapCard()` for each card individually, so each card gets its own undo-stack entry. A phrase like "ten six four" produces three undo entries. This is the current behavior and remains unchanged — saying "undo" reverses the last single card, not the whole phrase. This is acceptable since "undo" can be said repeatedly.

---

## Speech Grammar

User speaks numbers for pip cards and names for face cards:
- Pip cards: "2", "3", "4", "5", "6", "7", "8", "9", "10"
- Face cards: "jack", "queen", "king" (all map to T)
- Ace: "ace"
- Commands: "shuffle", "undo", "plus 3", "minus 2", "skip 5"

The existing `parseDictation()` already handles all these patterns plus noise words, digit-mash strings, and variations. No parser changes needed.

---

## UI Changes

### Removed
- Numpad grid (`.numpad` section — buttons for card ranks, adjustments)
- Text input + Apply button (`.voice-card` — the Wispr Flow input)
- Keymap hint card (keyboard shortcut reference)
- Keyboard shortcut listeners for card entry (numpad keys, rank letters)

### Kept Unchanged
- Bet hero panel (color-coded bet recommendation)
- Count strip (RC / TC / Decks left)
- Penetration bar with cut card indicator
- Active deviations grid
- Undo button, Shuffle button, Settings gear (actions row layout updated from 3-column to 4-column to accommodate mic toggle)

### New Elements

**Mic status indicator** — replaces the input area:
- Green pulsing dot + "Listening..." — mic active, Scribe connected
- Yellow + "Processing..." — transcript being committed
- Red + "Mic off" — disconnected or error, with error message
- Partial transcript text below (grey italic, updates live as Scribe processes, cleared when committed_transcript is received and action log replaces it)

**Transcript log** — below mic indicator:
- Shows last committed result: "Heard: 3 cards (RC +1)"
- Same format as current voice log

**API key field in setup wizard:**
- New text input: "ElevenLabs API Key"
- Stored in localStorage alongside game config (`setup.elevenLabsKey`)
- Entered once, persists across sessions

**Mic toggle button** — in actions row alongside Undo/Shuffle/Settings:
- Mute/unmute without leaving the page
- Cleanly closes/reopens audio stream + WebSocket

---

## Connection Lifecycle

### Startup
1. Setup screen → user enters game rules + API key → "Start session"
2. Browser prompts for mic permission
3. If granted → fetch single-use token from ElevenLabs REST API using stored API key
4. Open WebSocket to Scribe with token + config params
5. On `session_started` → mic indicator green, listening begins
6. If mic denied, token fetch fails, or WebSocket fails → indicator red with message, app still usable via buttons

### During Play
- WebSocket stays open for entire session
- Connection drop → auto-reconnect after 2s with exponential backoff (max 3 retries, caps at 8s)
- After 3 failed retries → "Connection lost — tap to reconnect" on mic indicator
- Mic toggle cleanly closes/reopens audio + WebSocket

### Error Handling
| Error Type | Response |
|------------|----------|
| `auth_error` | Show "Invalid API key" — prompt to re-enter in settings |
| `quota_exceeded` | Show "API quota exceeded" — disable mic |
| `session_time_limit_exceeded` | Fetch fresh token, then auto-reconnect (brief ~200ms gap) |
| `rate_limited` | Back off 5s, then reconnect |
| Network disconnect | Pause mic, show "Reconnecting...", fetch fresh token, reconnect with exponential backoff |
| Mic permission denied | Show "Mic access required" — settings link |

---

## What Does NOT Change

- `Lib` module — all pure functions (Hi-Lo counting, strategy, I18, Fab4, Wong ramp, parseDictation)
- `State` module — localStorage persistence, undo stack, shoe state
- `runTests()` — all 118+ unit tests
- Setup wizard — same fields plus one new API key input
- Game logic — counting, deviations, bet recommendations all identical

---

## File Changes

Only `bj/index.html` is modified:
- CSS: Remove numpad styles, add mic indicator styles, update `.actions-row` to 4-column grid
- HTML: Remove numpad/text-input sections, add mic indicator + transcript log + API key field
- JS/UI: Remove numpad event wiring + keyboard shortcuts for ranks, add Audio Capture (AudioWorklet via Blob URL) + WebSocket (with token fetch) + TTS modules

No new files created.
