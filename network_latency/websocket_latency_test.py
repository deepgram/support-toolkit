#!/usr/bin/env python3
"""
Deepgram WebSocket Latency Test
-------------------------------
Measures TCP/TLS latency, WebSocket handshake latency, and performs TCP traceroute.

Output: JSONL file (one JSON object per line) suitable for graphing.

Usage:
    uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3"
    uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3" --duration=60
    uv run websocket_latency_test.py "wss://api.deepgram.com/v1/listen?model=nova-3" --duration=60 --no-traceroute
"""

import argparse
import base64
import json
import os
import signal
import socket
import ssl
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


# ==============================
# CONFIGURATION
# ==============================
DEFAULT_OUTPUT_DIR = "results"
SLEEP_BETWEEN_ITERATIONS = 0.5
MAX_TRACEROUTE_HOPS = 30
TRACEROUTE_TIMEOUT = 30


# ==============================
# GLOBALS
# ==============================
stop_flag = False


def handle_signal(signum, frame):
    global stop_flag
    print(f"\nReceived signal {signum}. Stopping...")
    stop_flag = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ==============================
# DNS RESOLUTION
# ==============================
def resolve_dns(host: str, port: int) -> dict:
    """Resolve DNS and return IP address with timing."""
    start = time.perf_counter()
    try:
        addr_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ip_addresses = list(set(info[4][0] for info in addr_info))
        return {
            "status": "success",
            "dns_ms": elapsed_ms,
            "ip_address": ip_addresses[0],  # Use first IP
            "ip_addresses": ip_addresses,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status": "failed",
            "dns_ms": elapsed_ms,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


# ==============================
# WEBSOCKET HANDSHAKE
# ==============================
def measure_ws_handshake(host: str, port: int, ip_address: str, ws_path: str, api_key: str) -> dict:
    """Measure WebSocket handshake with detailed phase breakdown. Uses pre-resolved IP."""
    result = {"ip_address": ip_address}
    total_start = time.perf_counter()
    
    # TCP connection (using IP to avoid DNS)
    try:
        tcp_start = time.perf_counter()
        sock = socket.create_connection((ip_address, port), timeout=10)
        result["tcp_ms"] = (time.perf_counter() - tcp_start) * 1000
    except Exception as e:
        return {
            "status": "failed",
            "phase": "tcp",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
    
    # TLS handshake
    try:
        tls_start = time.perf_counter()
        context = ssl.create_default_context()
        ssock = context.wrap_socket(sock, server_hostname=host)
        result["tls_ms"] = (time.perf_counter() - tls_start) * 1000
    except Exception as e:
        sock.close()
        return {
            "status": "failed",
            "phase": "tls",
            "tcp_ms": result.get("tcp_ms"),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
    
    # WebSocket upgrade
    try:
        ws_key = base64.b64encode(os.urandom(16)).decode("utf-8")
        request = (
            f"GET {ws_path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Authorization: Token {api_key}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        
        ws_start = time.perf_counter()
        ssock.sendall(request.encode())
        
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = ssock.recv(1024)
            if not chunk:
                break
            response += chunk
        
        result["ws_upgrade_ms"] = (time.perf_counter() - ws_start) * 1000
        result["total_ms"] = (time.perf_counter() - total_start) * 1000
        
        ssock.close()
        
        # Parse response
        response_text = response.decode(errors="ignore")
        status_line = response_text.split("\r\n")[0] if response_text else ""
        
        # Extract HTTP status
        http_status = None
        if status_line.startswith("HTTP/"):
            parts = status_line.split(" ", 2)
            if len(parts) >= 2:
                try:
                    http_status = int(parts[1])
                except ValueError:
                    pass
        
        result["http_status"] = http_status
        
        # Extract dg-request-id
        for line in response_text.split("\r\n"):
            if line.lower().startswith("dg-request-id:"):
                result["dg_request_id"] = line.split(":", 1)[1].strip()
                break
        
        if http_status == 101:
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["phase"] = "ws_upgrade"
            result["error_type"] = "HTTPError"
            result["error_message"] = status_line
        
        return result
        
    except Exception as e:
        ssock.close()
        return {
            "status": "failed",
            "phase": "ws_upgrade",
            "tcp_ms": result.get("tcp_ms"),
            "tls_ms": result.get("tls_ms"),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


# ==============================
# TRACEROUTE
# ==============================
def run_traceroute(host: str, port: int = 443, max_hops: int = 30) -> dict:
    """Run TCP traceroute and parse results."""
    try:
        cmd = ["traceroute", "-T", "-p", str(port), "-m", str(max_hops), host]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TRACEROUTE_TIMEOUT
        )
        output = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "timeout"}
    except FileNotFoundError:
        return {"status": "failed", "error": "traceroute not installed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    
    # Parse traceroute output
    hops = []
    rtt_regex = re.compile(r"(\d+\.?\d*)\s*ms")
    hop_num_regex = re.compile(r"^\s*(\d+)\s+")
    
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("traceroute"):
            continue
        
        hop_match = hop_num_regex.match(line)
        hop_num = int(hop_match.group(1)) if hop_match else len(hops) + 1
        
        rtts = [float(m.group(1)) for m in rtt_regex.finditer(line)]
        avg_rtt = sum(rtts) / len(rtts) if rtts else None
        
        hops.append({
            "hop": hop_num,
            "rtt_ms": avg_rtt,
            "raw": line,
        })
    
    # Calculate summary stats
    rtt_values = [h["rtt_ms"] for h in hops if h["rtt_ms"] is not None]
    
    return {
        "status": "success",
        "hop_count": len(hops),
        "final_rtt_ms": rtt_values[-1] if rtt_values else None,
        "avg_rtt_ms": sum(rtt_values) / len(rtt_values) if rtt_values else None,
        "hops": hops,
    }


# ==============================
# MAIN TEST LOOP
# ==============================
def run_test(args):
    global stop_flag
    
    api_key = os.environ.get("DEEPGRAM_API_KEY") or args.api_key
    if not api_key:
        print("Error: DEEPGRAM_API_KEY not set. Use --api-key or set environment variable.")
        return 1
    
    host = args.host
    port = args.port
    ws_path = args.ws_path
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ws_latency_{host}.jsonl"
    
    print(f"WebSocket Latency Test")
    print(f"  Target: wss://{host}:{port}{ws_path}")
    print(f"  Output: {output_file}")
    if args.duration:
        print(f"  Duration: {args.duration}s")
    else:
        print(f"  Duration: indefinite (Ctrl+C to stop)")
    print(f"  Traceroute: {'disabled' if args.no_traceroute else 'enabled'}")
    print()
    
    start_time = time.time()
    iteration = 0
    successes = 0
    failures = 0
    
    with open(output_file, "a") as f:
        while not stop_flag:
            # Check duration limit
            elapsed = time.time() - start_time
            if args.duration and elapsed >= args.duration:
                break
            
            iteration += 1
            timestamp = datetime.now().isoformat()
            
            # Resolve DNS first
            dns_result = resolve_dns(host, port)
            
            # If DNS failed, we can't proceed with WebSocket
            if dns_result.get("status") == "failed":
                result = {
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "status": "failed",
                    "phase": "dns",
                    "dns": dns_result,
                }
                f.write(json.dumps(result) + "\n")
                f.flush()
                failures += 1
            else:
                # Measure WebSocket handshake using resolved IP
                ws_result = measure_ws_handshake(
                    host, port, dns_result["ip_address"], ws_path, api_key
                )
                
                # Run traceroute if enabled
                traceroute_result = None
                if not args.no_traceroute:
                    traceroute_result = run_traceroute(host, port, MAX_TRACEROUTE_HOPS)
                
                # Build result object
                result = {
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "status": ws_result.get("status", "failed"),
                    "dns": dns_result,
                    "websocket": ws_result,
                }
                
                if traceroute_result:
                    result["traceroute"] = traceroute_result
                
                # Write to file
                f.write(json.dumps(result) + "\n")
                f.flush()
                
                # Update counters
                if ws_result.get("status") == "success":
                    successes += 1
                else:
                    failures += 1
            
            # Progress output
            if args.duration:
                print(f"\rIteration {iteration} | Success: {successes} | Failed: {failures} | Elapsed: {elapsed:.1f}s / {args.duration}s", end="")
            else:
                print(f"\rIteration {iteration} | Success: {successes} | Failed: {failures} | Elapsed: {elapsed:.1f}s", end="")
            
            # Delay between iterations
            if args.delay > 0:
                time.sleep(args.delay)
    
    print(f"\n\nDone. {successes} successful, {failures} failed.")
    print(f"Results: {output_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Deepgram WebSocket Latency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run %(prog)s "wss://api.deepgram.com/v1/listen?model=nova-3"
  uv run %(prog)s "wss://api.deepgram.com/v1/listen?model=nova-3" --duration=60
  uv run %(prog)s "wss://api.deepgram.com/v1/listen?model=nova-3" --no-traceroute
        """,
    )
    
    parser.add_argument(
        "url",
        help="WebSocket URL (wss://...)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Run for specified seconds (default: run indefinitely)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=SLEEP_BETWEEN_ITERATIONS,
        help=f"Delay between iterations in seconds (default: {SLEEP_BETWEEN_ITERATIONS})",
    )
    parser.add_argument(
        "--no-traceroute",
        action="store_true",
        help="Disable traceroute (faster sample collection)",
    )
    parser.add_argument(
        "--api-key",
        help="Deepgram API key (default: uses DEEPGRAM_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    
    args = parser.parse_args()
    
    # Parse the URL
    url_input = args.url
    
    if not (url_input.startswith("wss://") or url_input.startswith("ws://")):
        parser.error(f"URL must start with wss:// or ws://")
    
    parsed = urlparse(url_input)
    args.host = parsed.hostname
    args.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    args.ws_path = parsed.path
    if parsed.query:
        args.ws_path += "?" + parsed.query
    
    return run_test(args)


if __name__ == "__main__":
    exit(main())
