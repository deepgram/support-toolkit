# WebSocket Latency Testing Tools

Tools for measuring WebSocket connection latency to Deepgram's API.

## Requirements

- [uv](https://github.com/astral-sh/uv)
- `traceroute` command (Linux/Mac) - optional, for network path analysis

## Usage

### 1. Collect Data

```bash
uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3"
```

Options:
- `--duration=60` - Run for 60 seconds (default: indefinite, Ctrl+C to stop)
- `--no-traceroute` - Disable traceroute for faster collection
- `--delay=0.5` - Delay between iterations in seconds (default: 0.5)
- `--api-key=KEY` - API key (default: `DEEPGRAM_API_KEY` env var)
- `--output-dir=DIR` - Output directory (default: `results`)

Output: `results/ws_latency_<hostname>.jsonl`

### 2. Analyze Results

```bash
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl
```

Options:
- `-o latency.png` - Save graphs to file (default: display interactively)
- `--time-series` - Also generate latency-over-time plot
- `--summary-only` - Print statistics only, no graphs
- `--no-explanation` - Skip detailed metric explanations

## Example

```bash
# Collect 60 seconds of data without traceroute
uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3" --duration=60 --no-traceroute

# Generate graphs
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl -o latency.png
```

## What It Measures

| Metric | Description | Indicates |
|--------|-------------|-----------|
| DNS | Domain name resolution | Client-side (DNS resolver) |
| TCP | Network connection establishment | Network path / distance |
| TLS | Encryption negotiation | Network + some server |
| WebSocket Upgrade | HTTP 101 handshake | Server processing |
| Traceroute | Hop-by-hop network path | Where latency occurs |
