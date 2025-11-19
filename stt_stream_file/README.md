# Deepgram Audio Streaming & Transcription Tools

Real-time audio transcription with a beautiful terminal UI. Stream from your microphone or audio files and see transcripts appear instantly with color-coded confidence scores, speaker labels, and latency metrics.

## Quick Start
```bash
pip install -r requirements.txt
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

**Stream to file (realtime mode - natural pace):**
```bash
uv run stream_audio_file.py -o output.json -f audio.wav --realtime \
  --url "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true"
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
  --print-speakers --print-channels --print-interim --print-delay --colorize
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
- `--ui` - Interactive terminal UI with live updates
- `-f, --audio` - Audio file to stream
- `-l, --live` - Stream from microphone
- `-o, --output` - Save JSON messages to file
- `-v, -vv, -vvv` - Increase verbosity

### print_transcript.py
- `--print-speakers` - Show speaker labels
- `--print-channels` - Show audio channels
- `--print-interim` - Include interim results
- `--print-delay` - Show latency (time since end of speech)
- `--colorize` - Color words by confidence
- `--only-transcript` - Just the text, no metadata

Run either script with `--help` for full options.

## What's Happening?

The UI mode shows transcription speed in real-time - watch words appear as you speak and see exactly how fast Deepgram processes your audio. The `--print-delay` option reveals latency metrics, perfect for testing different models and configurations.


