# WebSocket Latency Testing Tools

Tools for measuring WebSocket connection latency to Deepgram's API.

## Requirements

- [uv](https://github.com/astral-sh/uv)
- `traceroute` command (Linux/Mac) - optional, for network path analysis

## Usage

There are **two distinct modes** for different purposes:

### Mode 1: Statistical Analysis (Recommended for most users)

**Purpose**: Collect hundreds of samples quickly to analyze latency distribution, identify patterns, and detect performance issues.

**When to use**: 
- Initial performance assessment
- Comparing latency across different times of day
- Building statistical profiles (min/max/p95/p99)
- Monitoring performance over time

**How to run**:
```bash
# Collect 10 minutes of data (fast sampling, no traceroute)
uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3" --duration=600 --no-traceroute
```

**Key characteristics**:
- Use `--duration=X` to run for X seconds (e.g., 600 = 10 minutes)
- **Always use `--no-traceroute`** for fast sampling (~0.5s per iteration)
- Collects 100+ samples in minutes
- Ideal for generating statistical graphs

### Mode 2: Deep Network Diagnostics

**Purpose**: Detailed hop-by-hop network path analysis to diagnose routing issues or understand where latency occurs.

**When to use**:
- Troubleshooting unexpected high latency
- Understanding network routing to Deepgram servers
- Investigating intermittent connection issues
- Analyzing specific network path problems

**How to run**:
```bash
# Run indefinitely with traceroute (stop with Ctrl+C)
uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3"
```

**Key characteristics**:
- Run **without** `--duration` (stops with Ctrl+C)
- Traceroute enabled by default
- **Much slower**: ~30+ seconds per iteration (traceroute overhead)
- Collects ~10-20 samples in 10 minutes
- Shows hop-by-hop network path to identify routing issues

### Analyzing Results

After collecting data with either mode:

```bash
# View interactive graphs
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl

# Save graphs to file
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl -o latency.png

# Include time-series plot
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl --time-series

# Statistics only (no graphs)
uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl --summary-only
```

### Common Options

Both modes support:
- `--api-key=KEY` - API key (default: `DEEPGRAM_API_KEY` env var)
- `--output-dir=DIR` - Output directory (default: `results`)
- `--delay=X` - Delay between iterations in seconds (default: 0.5)

Output file: `results/ws_latency_<hostname>.jsonl`

## What It Measures

| Metric | Description | Indicates |
|--------|-------------|-----------|
| DNS | Domain name resolution | Client-side (DNS resolver) |
| TCP | Network connection establishment | Network path / distance |
| TLS | Encryption negotiation | Network + some server |
| WebSocket Upgrade | HTTP 101 handshake | Server processing |
| Traceroute | Hop-by-hop network path | Where latency occurs |

## Understanding Results

This section explains each metric, what's normal, and how to diagnose issues.

To check your baseline network latency to Deepgram's servers (some servers are located in San Francisco), you can use: https://wondernetwork.com/pings/San%20Francisco

### Metrics Explained

**1. DNS Resolution**

What it measures: Time to convert the domain name (e.g., api.deepgram.com) into an IP address.

What's normal: 1-50ms. Often <5ms if cached locally.

If this is slow: Your DNS resolver may be slow or overloaded. Consider:
- Using a faster DNS provider (8.8.8.8, 1.1.1.1)
- Checking if your local DNS cache is working

This is purely client-side — Deepgram has no control over this.

**2. TCP Connection**

What it measures: Time to establish a basic network connection to the server (the "three-way handshake": SYN, SYN-ACK, ACK).

What's normal: Depends on your geographic distance to Deepgram's servers (San Francisco):
- Same city/region: 5-20ms
- Same continent (e.g., US East Coast): 60-100ms
- Cross-continental (e.g., Europe to US): 120-180ms
- Intercontinental (e.g., Asia/Australia to US): 150-300ms

Check your expected latency: https://wondernetwork.com/pings/San%20Francisco

This should be very consistent across measurements. High variance indicates network issues.

If this is slow or variable: Indicates network path issues:
- Geographic distance to server
- Network congestion
- Packet loss causing retransmissions
- ISP routing issues

This reflects the network path between you and Deepgram's servers.

**3. TLS Handshake**

What it measures: Time to negotiate encryption after the TCP connection is established. This involves exchanging certificates and cryptographic keys.

What's normal: Typically 1-2x your TCP time (1-2 network round trips). TLS 1.3 requires fewer round trips than TLS 1.2.
- Same continent: 20-100ms
- Cross-continental: 120-250ms
- Intercontinental: 150-400ms

If this is slow: Could indicate:
- Network latency (TLS requires multiple round trips)
- Server CPU load during certificate operations
- Certificate chain validation issues on the client
- Older TLS versions requiring more round trips

Mostly network-dependent, with some server-side component.

**4. WebSocket Upgrade**

What it measures: Time from sending the HTTP upgrade request to receiving the "101 Switching Protocols" response. This is when Deepgram's servers process your connection request and authenticate your API key.

What's normal: Your TCP time (1 RTT) plus server processing time. Server processing typically adds 50-150ms. Total:
- Same continent: 70-200ms
- Cross-continental: 170-350ms
- Intercontinental: 200-450ms

If this is slow: Could indicate:
- Network latency (at least one round trip required)
- Server-side processing delays (authentication, resource allocation)
- High server load during peak times

This is where server-side processing time shows up most clearly. If TCP and TLS are normal but WebSocket upgrade is slow, the issue is likely on the server side. Server processing typically requires 100-200ms, so if your websocket upgrade statistics are significantly longer than (the TCP Connection time + 200ms) then additional investigation may be required.

**5. Total Connection Time**

The sum of all phases above. This is the complete time from starting the request to having a fully established, authenticated WebSocket connection ready to send/receive audio data.

**6. Traceroute Final Hop RTT (if collected)**

What it measures: Round-trip time to Deepgram's servers as measured by traceroute. Useful for understanding the network path.

If significantly different from TCP time: May indicate routing changes or network instability.

### Understanding the Statistics

- **Min**: Fastest measurement (best case)
- **Max**: Slowest measurement (worst case) — check for outliers
- **Mean**: Average — useful for overall picture
- **Median**: Middle value — less affected by outliers than mean
- **P95**: 95th percentile — 5% of requests were slower than this
- **P99**: 99th percentile — 1% of requests were slower than this

If Mean >> Median: You have outliers pulling the average up. Look at P95/P99.

If P95 >> Median: Occasional slow requests. May indicate intermittent issues.

### Understanding the Histograms

- X-axis: Time in milliseconds
- Y-axis: How many measurements fell into each time bucket
- Red dashed line: Mean (average)
- Green dotted line: Median (middle value)

What to look for:
- Tight, narrow distribution = consistent performance (good)
- Wide or spread out = variable performance (investigate)
- Multiple peaks = possible different network paths or server behavior (Deepgram has two server locations in the US so two peaks are expected when measuring traffic to the US)
- Long tail to the right = occasional slow requests

### Diagnosing Issues

**High DNS latency?**
→ Client-side issue. Check your DNS resolver, consider 8.8.8.8 or 1.1.1.1

**High TCP latency?**
→ Network path issue. Compare your TCP time to the expected ping from https://wondernetwork.com/pings/San%20Francisco for your region. If much higher, check ISP routing or consider a VPN/different network path.

**High TLS latency?**
→ Mostly network (multiple round trips). Some server component. If much higher than 2x TCP, may indicate certificate issues.

**High WebSocket Upgrade latency?**
→ Server-side processing or network. If TCP/TLS are normal but WS upgrade is slow, contact Deepgram support and share the results produced by these scripts.

**Variable latency (high P95 vs median)?**
→ Intermittent network issues or server load. Collect data over longer period to identify patterns (time of day, etc.)
