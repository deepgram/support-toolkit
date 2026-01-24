# Deepgram Audio Streaming & Transcription Tools

Real-time audio transcription with a beautiful terminal UI. Stream from your microphone or audio files and see transcripts appear instantly with color-coded confidence scores, speaker labels, and latency metrics.

## Quick Start
```bash
uv sync
export DEEPGRAM_API_KEY="your_api_key_here"
```

Then launch the UI and start talking:
```bash
uv run stream_audio_file.py --ui --live \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&interim_results=true"
```

## Examples

### Interactive UI Mode

**Live microphone streaming:**
```bash
uv run stream_audio_file.py --ui --live \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&interim_results=true"
```

**Transcribe an audio file:**
```bash
uv run stream_audio_file.py --ui -f audio.wav \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
```

**Try Flux for ultra-low latency:**
```bash
uv run stream_audio_file.py --ui --live \
  --url "wss://api.deepgram.com/v2/listen?model=flux-general-en&eot_threshold=0.7&encoding=linear16&sample_rate=16000"
```

### Save & Print Mode

**Stream and save JSON output:**
```bash
uv run stream_audio_file.py -f audio.wav \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true"
```

Output is automatically saved to `audio.json` (derived from input filename).

**Specify a custom output file:**
```bash
uv run stream_audio_file.py -o output.json -f audio.wav \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true"
```

**Live recording saves with timestamp:**
```bash
uv run stream_audio_file.py --live \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true"
# Saves to recording_20250114_153022.json (or similar)
```

**Print basic transcript:**
```bash
uv run print_transcript.py -f output.json
```
```
[00:00:00.56 - 00:00:03.29]: The missile knows where it is at all times.
[00:00:03.75 - 00:00:06.17]: It knows this because it knows where it isn't.
```

**Print with all the details:**
```bash
uv run print_transcript.py -f output.json \
  --print-speakers --print-channels --print-interim --print-latency --colorize
```
```
[18:30:24.066 (0.665s since EOS)] [00:00:00.00 - 00:00:03.48] [Speaker 0] [Channel 0] [IsFinal]: The missile knows where it is at all times.
```

**Print just the text:**
```bash
uv run print_transcript.py -f output.json --only-transcript
```
```
The missile knows where it is at all times.
It knows this because it knows where it isn't.
```

## Key Options

### stream_audio_file.py

| Option | Description |
|--------|-------------|
| `--url, -u` | Deepgram websocket URL (required) |
| `--ui` | Interactive terminal UI with live updates |
| `-f, --audio` | Audio file to stream |
| `-l, --live` | Stream from microphone |
| `-o, --output` | Save JSON messages to file (defaults to input filename or timestamped name) |
| `-v, -vv, -vvv` | Increase verbosity |

### print_transcript.py

| Option | Description |
|--------|-------------|
| `--print-speakers` | Show speaker labels |
| `--print-channels` | Show audio channels |
| `--print-interim` | Include interim results |
| `--print-received` | Show received timestamp for streamed messages |
| `--print-latency` | Show latency metrics (TTFT, update frequency, message latency, EOT latency) |
| `--print-entities` | Show detected entities |
| `--colorize` | Color words by confidence |
| `--only-transcript` | Just the text, no metadata |

Run either script with `--help` for full options.

### Shell Completion

Generate shell completions for your preferred shell:

```bash
uv run stream_audio_file.py completion bash  # or zsh, fish
```

## Metrics Calculated

When using `--print-latency`, the following metrics are computed:

**Session-level:**
- **TTFT (Time To First Transcript)**: Wall-clock time from when audio streaming begins to when the first transcript message is received. Measures initial responsiveness.
- **Update Frequency**: Number of interim transcript updates per second of audio. Higher values mean a more fluid, responsive transcription experience.

**Per-message:**
- **Message Latency**: How far behind the transcription is from the audio being sent, calculated as `audio_cursor - transcript_cursor`. Measured on interim results only, per Deepgram's methodology.
- **EOT Latency (End-of-Turn Latency)**: Time between the last interim result and the finalizing event (e.g., `speech_final`, `UtteranceEnd`, `EndOfTurn`). Critical for voice agents—they can't respond until they know the user stopped speaking.

## What's Happening?

The UI mode shows transcription speed in real-time—watch words appear as you speak and see exactly how fast Deepgram processes your audio. The `--print-latency` option reveals latency metrics, perfect for testing different models and configurations.