# Deepgram TTS Latency Measurement

Stream text to Deepgram's TTS websocket API and record detailed timing metrics for latency analysis.

## Setup

```bash
# Install dependencies with uv
uv sync

# Set your API key
export DEEPGRAM_API_KEY="your-api-key"
```

## Usage

### 1. Stream TTS and Collect Timing Data

```bash
# Basic usage
uv run stream_tts.py -i phrases.txt

# With custom output paths
uv run stream_tts.py -i phrases.txt -o results.json -a output.wav

# With verbose output
uv run stream_tts.py -i phrases.txt -vv

# With custom model
uv run stream_tts.py -i phrases.txt -m aura-2-asteria-en
```

### 2. Analyze Latency Metrics

```bash
# Basic analysis (prints human-readable report)
uv run analyze_tts_latency.py -i results.json

# With per-phrase details
uv run analyze_tts_latency.py -i results.json -v

# With per-packet details
uv run analyze_tts_latency.py -i results.json -vv

# Export computed metrics to JSON
uv run analyze_tts_latency.py -i results.json -o metrics.json

# JSON output only (no report)
uv run analyze_tts_latency.py -i results.json --json-only
```

## Input File Format

One phrase per line. Empty lines and lines starting with `#` or `//` are ignored.

```text
# This is a comment
Hello, this is the first phrase.
The second phrase goes here.
```

## Metrics Calculated

**Session-level:**
- **TTFB (Time To First Byte)**: Time from first text sent to first audio byte received
- **TTFB incl. network**: Time from session start (before websocket connection) to first audio byte received
- **TTLB (Time To Last Byte)**: Time from first text sent to last audio byte received
- **Overall RTF**: total_audio_duration / delivery_time
- **Min Cumulative RTF**: The lowest point of cumulative_audio / wall_clock across all packets. If >= 1.0, the stream never fell behind real-time playback.

**Per-phrase:**
- **Audio duration**: Total duration of audio content in the phrase
- **Delivery time**: Wall-clock time from first to last packet of the phrase
- **RTF**: audio_duration / delivery_time
- **Min Cumulative RTF**: Lowest cumulative RTF within the phrase
- **Jitter**: Standard deviation of inter-arrival times within the phrase

## Output Files

- **JSON timing file**: Raw timing data for each phrase and packet
- **WAV audio file**: The synthesized audio
- **Metrics JSON** (optional): Computed latency statistics
