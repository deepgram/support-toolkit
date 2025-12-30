#!/usr/bin/env python3
"""
Graph WebSocket latency test results from JSONL output.

Usage:
    uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl
    uv run --with matplotlib websocket_latency_graph.py results/ws_latency_api.deepgram.com.jsonl -o latency.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(filepath: Path) -> tuple[list[dict], list[dict]]:
    """Load results from JSONL file, separating successes and failures."""
    successes = []
    failures = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("status") == "success":
                    successes.append(entry)
                else:
                    failures.append(entry)
    return successes, failures


def extract_metrics(results: list[dict]) -> dict[str, list[float]]:
    """Extract timing metrics from results."""
    metrics = {
        "dns_ms": [],
        "tcp_ms": [],
        "tls_ms": [],
        "ws_upgrade_ms": [],
        "total_ms": [],
        "traceroute_final_rtt_ms": [],
    }
    
    for r in results:
        # DNS
        if r.get("dns", {}).get("status") == "success":
            dns_ms = r["dns"].get("dns_ms")
            if dns_ms is not None:
                metrics["dns_ms"].append(dns_ms)
        
        # WebSocket phases
        ws = r.get("websocket", {})
        if ws.get("status") == "success":
            if ws.get("tcp_ms") is not None:
                metrics["tcp_ms"].append(ws["tcp_ms"])
            if ws.get("tls_ms") is not None:
                metrics["tls_ms"].append(ws["tls_ms"])
            if ws.get("ws_upgrade_ms") is not None:
                metrics["ws_upgrade_ms"].append(ws["ws_upgrade_ms"])
            if ws.get("total_ms") is not None:
                metrics["total_ms"].append(ws["total_ms"])
        
        # Traceroute
        tr = r.get("traceroute", {})
        if tr.get("status") == "success" and tr.get("final_rtt_ms") is not None:
            metrics["traceroute_final_rtt_ms"].append(tr["final_rtt_ms"])
    
    return metrics


def calc_stats(data: list[float]) -> dict:
    """Calculate statistics for a list of values."""
    if not data:
        return {"count": 0}
    
    sorted_data = sorted(data)
    n = len(data)
    
    return {
        "count": n,
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / n,
        "median": sorted_data[n // 2],
        "p95": sorted_data[int(n * 0.95)] if n >= 20 else sorted_data[-1],
        "p99": sorted_data[int(n * 0.99)] if n >= 100 else sorted_data[-1],
    }


def print_explanations() -> None:
    """Print customer-friendly explanations of each metric."""
    print("""
================================================================================
                   WEBSOCKET LATENCY EXPLANATION
================================================================================

This report shows how long each step of establishing a WebSocket connection
takes. Understanding these metrics helps identify where latency issues originate.

--------------------------------------------------------------------------------
METRICS EXPLAINED
--------------------------------------------------------------------------------

1. DNS Resolution
   What it measures: Time to convert the domain name (e.g., api.deepgram.com)
   into an IP address.
   
   What's normal: 1-50ms. Often <5ms if cached locally.
   
   If this is slow: Your DNS resolver may be slow or overloaded. Consider:
   - Using a faster DNS provider (8.8.8.8, 1.1.1.1)
   - Checking if your local DNS cache is working
   
   This is purely client-side — Deepgram has no control over this.

2. TCP Connection
   What it measures: Time to establish a basic network connection to the server
   (the "three-way handshake": SYN, SYN-ACK, ACK).
   
   What's normal: 10-100ms depending on geographic distance. Should be very
   consistent across measurements.
   
   If this is slow or variable: Indicates network path issues:
   - Geographic distance to server
   - Network congestion
   - Packet loss causing retransmissions
   - ISP routing issues
   
   This reflects the network path between you and Deepgram's servers.

3. TLS Handshake
   What it measures: Time to negotiate encryption after the TCP connection is
   established. This involves exchanging certificates and cryptographic keys.
   
   What's normal: 20-100ms. Typically requires 1-2 network round trips.
   
   If this is slow: Could indicate:
   - Network latency (TLS requires multiple round trips)
   - Server CPU load during certificate operations
   - Certificate chain validation issues on the client
   - Older TLS versions requiring more round trips
   
   Mostly network-dependent, with some server-side component.

4. WebSocket Upgrade
   What it measures: Time from sending the HTTP upgrade request to receiving
   the "101 Switching Protocols" response. This is when Deepgram's servers
   process your connection request and authenticate your API key.
   
   What's normal: 20-150ms. Includes one network round trip plus server
   processing time.
   
   If this is slow: Could indicate:
   - Network latency (at least one round trip required)
   - Server-side processing delays (authentication, resource allocation)
   - High server load during peak times
   
   This is where server-side processing time shows up most clearly.
   If TCP and TLS are normal but WebSocket upgrade is slow, the issue
   is likely on the server side.

5. Total Connection Time
   The sum of all phases above. This is the complete time from starting the
   request to having a fully established, authenticated WebSocket connection
   ready to send/receive audio data.

6. Traceroute Final Hop RTT (if collected)
   What it measures: Round-trip time to Deepgram's servers as measured by
   traceroute. Useful for understanding the network path.
   
   If significantly different from TCP time: May indicate routing changes
   or network instability.

--------------------------------------------------------------------------------
UNDERSTANDING THE STATISTICS
--------------------------------------------------------------------------------

- Min:    Fastest measurement (best case)
- Max:    Slowest measurement (worst case) — check for outliers
- Mean:   Average — useful for overall picture
- Median: Middle value — less affected by outliers than mean
- P95:    95th percentile — 95% of requests were faster than this
- P99:    99th percentile — only 1% of requests were slower

If Mean >> Median: You have outliers pulling the average up. Look at P95/P99.
If P95 >> Median: Occasional slow requests. May indicate intermittent issues.

--------------------------------------------------------------------------------
UNDERSTANDING THE HISTOGRAMS
--------------------------------------------------------------------------------

- X-axis: Time in milliseconds
- Y-axis: How many measurements fell into each time bucket
- Red dashed line: Mean (average)
- Green dotted line: Median (middle value)

What to look for:
- Tight, narrow distribution = consistent performance (good)
- Wide or spread out = variable performance (investigate)
- Multiple peaks = possible different network paths or server behavior
- Long tail to the right = occasional slow requests

--------------------------------------------------------------------------------
DIAGNOSING ISSUES
--------------------------------------------------------------------------------

High DNS latency?
  → Client-side issue. Check your DNS resolver, consider 8.8.8.8 or 1.1.1.1

High TCP latency?
  → Network path issue. Geographic distance, ISP routing, or congestion.
    Consider if a closer Deepgram region is available.

High TLS latency?
  → Mostly network (multiple round trips). Some server component.
    If much higher than 2x TCP, may indicate certificate issues.

High WebSocket Upgrade latency?
  → Server-side processing or network. If TCP/TLS are normal but WS upgrade
    is slow, contact Deepgram support with your dg-request-id values.

Variable latency (high P95 vs median)?
  → Intermittent network issues or server load. Collect data over longer
    period to identify patterns (time of day, etc.)

================================================================================
""")


def print_summary(metrics: dict[str, list[float]]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    labels = {
        "dns_ms": "DNS Resolution",
        "tcp_ms": "TCP Connection",
        "tls_ms": "TLS Handshake",
        "ws_upgrade_ms": "WebSocket Upgrade",
        "total_ms": "Total Connection",
        "traceroute_final_rtt_ms": "Traceroute Final Hop",
    }
    
    for key, label in labels.items():
        data = metrics.get(key, [])
        stats = calc_stats(data)
        
        if stats["count"] == 0:
            continue
        
        print(f"\n{label} (n={stats['count']})")
        print(f"  Min:    {stats['min']:8.2f} ms")
        print(f"  Max:    {stats['max']:8.2f} ms")
        print(f"  Mean:   {stats['mean']:8.2f} ms")
        print(f"  Median: {stats['median']:8.2f} ms")
        print(f"  P95:    {stats['p95']:8.2f} ms")
    
    print("\n" + "=" * 70)


def create_graphs(metrics: dict[str, list[float]], output_path: Path | None) -> None:
    """Create histogram graphs for each metric."""
    
    # Filter to metrics with data
    plot_configs = [
        ("dns_ms", "DNS Resolution"),
        ("tcp_ms", "TCP Connection"),
        ("tls_ms", "TLS Handshake"),
        ("ws_upgrade_ms", "WebSocket Upgrade"),
        ("total_ms", "Total Connection Time"),
        ("traceroute_final_rtt_ms", "Traceroute Final Hop RTT"),
    ]
    
    available = [(key, label) for key, label in plot_configs if metrics.get(key)]
    
    if not available:
        print("No data to graph.")
        return
    
    # Determine grid size
    n_plots = len(available)
    if n_plots <= 2:
        rows, cols = 1, n_plots
    elif n_plots <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    total_samples = max(len(v) for v in metrics.values()) if metrics else 0
    fig.suptitle(f"WebSocket Latency Distribution (n={total_samples} samples)", fontsize=14)
    
    for idx, (key, label) in enumerate(available):
        ax = axes[idx]
        data = metrics[key]
        stats = calc_stats(data)
        
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(stats["mean"], color="red", linestyle="--", 
                   label=f"Mean: {stats['mean']:.1f}ms")
        ax.axvline(stats["median"], color="green", linestyle=":", 
                   label=f"Median: {stats['median']:.1f}ms")
        
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency")
        ax.set_title(label)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Graph saved to: {output_path}")
    else:
        plt.show()


def create_time_series(results: list[dict], output_path: Path | None) -> None:
    """Create time series plot showing latency over time."""
    
    timestamps = []
    tcp_ms = []
    tls_ms = []
    ws_upgrade_ms = []
    
    for r in results:
        ws = r.get("websocket", {})
        if ws.get("status") == "success":
            timestamps.append(r.get("iteration", len(timestamps) + 1))
            tcp_ms.append(ws.get("tcp_ms", 0))
            tls_ms.append(ws.get("tls_ms", 0))
            ws_upgrade_ms.append(ws.get("ws_upgrade_ms", 0))
    
    if not timestamps:
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(timestamps, tcp_ms, label="TCP", alpha=0.7)
    ax.plot(timestamps, tls_ms, label="TLS", alpha=0.7)
    ax.plot(timestamps, ws_upgrade_ms, label="WS Upgrade", alpha=0.7)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        ts_path = output_path.with_stem(output_path.stem + "_timeseries")
        plt.savefig(ts_path, dpi=150)
        print(f"Time series saved to: {ts_path}")
    else:
        plt.show()


def print_error_summary(errors: list[dict]) -> None:
    """Print summary of errors."""
    if not errors:
        return
    
    print("\n" + "=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)
    
    # Group by phase
    by_phase = {}
    for e in errors:
        # Determine phase from the error structure
        phase = e.get("phase", "unknown")
        if phase == "unknown":
            # Check websocket result for phase
            ws = e.get("websocket", {})
            if ws.get("status") == "failed":
                phase = ws.get("phase", "unknown")
        
        if phase not in by_phase:
            by_phase[phase] = []
        by_phase[phase].append(e)
    
    print(f"\nTotal errors: {len(errors)}")
    print("\nErrors by phase:")
    for phase, phase_errors in sorted(by_phase.items()):
        print(f"  {phase}: {len(phase_errors)}")
        
        # Group by error type within phase
        by_type = {}
        for e in phase_errors:
            # Get error type from appropriate location
            if phase == "dns":
                error_type = e.get("dns", {}).get("error_type", "unknown")
                error_message = e.get("dns", {}).get("error_message", "")
            else:
                ws = e.get("websocket", {})
                error_type = ws.get("error_type", "unknown")
                error_message = ws.get("error_message", "")
            
            if error_type not in by_type:
                by_type[error_type] = []
            by_type[error_type].append({"error_message": error_message, "entry": e})
        
        for error_type, type_errors in sorted(by_type.items()):
            print(f"    {error_type}: {len(type_errors)}")
            # Show a sample error message
            sample = type_errors[0].get("error_message", "")
            if sample:
                # Truncate long messages
                if len(sample) > 60:
                    sample = sample[:60] + "..."
                print(f"      Example: {sample}")
    
    # Check for specific patterns
    print("\n" + "-" * 70)
    print("DIAGNOSIS")
    print("-" * 70)
    
    dns_errors = len(by_phase.get("dns", []))
    tcp_errors = len(by_phase.get("tcp", []))
    tls_errors = len(by_phase.get("tls", []))
    ws_errors = len(by_phase.get("ws_upgrade", []))
    
    if dns_errors > 0:
        print(f"\n⚠ DNS errors ({dns_errors}): Check your DNS resolver configuration")
    
    if tcp_errors > 0:
        print(f"\n⚠ TCP errors ({tcp_errors}): Network connectivity issues")
        print("  - Check firewall settings")
        print("  - Verify outbound connections to port 443 are allowed")
    
    if tls_errors > 0:
        print(f"\n⚠ TLS errors ({tls_errors}): SSL/TLS handshake failures")
        print("  - Check system clock is accurate")
        print("  - Verify CA certificates are up to date")
    
    if ws_errors > 0:
        # Check for auth errors
        auth_errors = sum(1 for e in by_phase.get("ws_upgrade", []) 
                        if e.get("websocket", {}).get("http_status") in (401, 403))
        if auth_errors:
            print(f"\n⚠ Authentication errors ({auth_errors}): Check your API key")
        else:
            print(f"\n⚠ WebSocket upgrade errors ({ws_errors}): Server rejected connection")
            print("  - Check the WebSocket path is correct")
            print("  - Verify API key has appropriate permissions")


def main():
    parser = argparse.ArgumentParser(
        description="Graph WebSocket latency test results"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file from websocket_latency_test.py",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output image file (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="Also generate time series plot",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary statistics only, no graphs",
    )
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Skip the detailed explanation of metrics",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    
    # Load data
    results, errors = load_results(args.input)
    total = len(results) + len(errors)
    
    if total == 0:
        print("Error: No results found in file", file=sys.stderr)
        return 1
    
    print(f"Loaded {total} results from {args.input}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(errors)}")
    
    # Extract metrics from successful results
    metrics = extract_metrics(results)
    
    # Print explanation (unless suppressed)
    if not args.no_explanation:
        print_explanations()
    
    # Print summary
    print_summary(metrics)
    
    # Print error summary
    if errors:
        print_error_summary(errors)
    
    if args.summary_only:
        return 0
    
    # Create graphs
    create_graphs(metrics, args.output)
    
    if args.time_series:
        create_time_series(results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
