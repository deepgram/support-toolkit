#!/usr/bin/env python3
"""
Deepgram WebSocket Latency Test + TCP Traceroute per Iteration (Long-Running Safe)
-------------------------------------------------------------------------------
1. Measures TCP/TLS latency and WebSocket handshake latency.
2. Extracts IP addresses from DNS resolution.
3. Extracts dg-request-id from handshake.
4. Flags high latency (>1000ms).
5. Performs TCP traceroute to target on port 443 each iteration.
6. Writes results immediately to a JSONL file (one JSON per line).
7. Breaks down potential client-side vs server-side latency issues.
8. Generates summary with min, max, and average latencies for each metric.
"""

import socket
import ssl
import time
import base64
import signal
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import re
import argparse

# ==============================
# CONFIGURATION
# ==============================
API_KEY = os.getenv("DEEPGRAM_API_KEY") 
DEEPGRAM_HOST = "api.deepgram.com"
DEEPGRAM_PORT = 443
FLUX_WS_PATH = "/v2/listen?model=flux-general-en"
NOVA2_WS_PATH = "/v1/listen?model=nova-2"
NOVA3_WS_PATH = "/v1/listen?model=nova-3"

# Model to WS path mapping
MODEL_PATHS = {
    "flux": FLUX_WS_PATH,
    "nova-2": NOVA2_WS_PATH,
    "nova-3": NOVA3_WS_PATH,
}

OUTPUT_DIR = "deepgram_results"
RESULTS_FILE = "ws_latency_results.jsonl"
SUMMARY_FILE = "ws_latency_summary.json"
LOG_FILE = "ws_test_log.txt"
SLEEP_BETWEEN_ITERATIONS = 2
HIGH_LATENCY_THRESHOLD_MS = 1000
MAX_TRACEROUTE_HOPS = 30
TRACEROUTE_TIMEOUT = 30  # seconds per iteration

# ==============================
# GLOBALS
# ==============================
iteration = 1
stop_flag = False

# ==============================
# SIGNAL HANDLER
# ==============================
def handle_signal(signum, frame):
    global stop_flag
    print(f"\n🛑 Received signal {signum}. Stopping test...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ==============================
# UTILITIES
# ==============================
def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def write_log(message, model_name=None):
    log_file = f"ws_test_log_{model_name}.txt" if model_name else LOG_FILE
    with open(Path(OUTPUT_DIR) / log_file, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {message}\n")

def append_result_to_file(result, model_name=None):
    results_file = f"ws_latency_results_{model_name}.jsonl" if model_name else RESULTS_FILE
    with open(Path(OUTPUT_DIR) / results_file, "a") as f:
        f.write(json.dumps(result) + "\n")

def fmt_ms(v):
    """Format a value as milliseconds to 2 decimals if numeric; else return as string."""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)

# ==============================
# DNS RESOLUTION 
# ==============================
def test_dns_resolution():
    """Measure DNS lookup time - indicates client-side DNS issues"""
    start = time.perf_counter()
    try:
        addr_info = socket.getaddrinfo(DEEPGRAM_HOST, DEEPGRAM_PORT, socket.AF_INET, socket.SOCK_STREAM)
        dns_ms = (time.perf_counter() - start) * 1000
        # Extract IP addresses from getaddrinfo results
        ip_addresses = list(set([info[4][0] for info in addr_info]))
        return {"status": "success", "dns_ms": dns_ms, "ip_addresses": ip_addresses}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"status": "failed", "error": str(e), "dns_ms": elapsed_ms, "ip_addresses": []}

def test_tcp_latency():
    start = time.perf_counter()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((DEEPGRAM_HOST, DEEPGRAM_PORT), timeout=20) as sock:
            with context.wrap_socket(sock, server_hostname=DEEPGRAM_HOST) as ssock:
                latency_ms = (time.perf_counter() - start) * 1000
                return {"status": "success", "connection_ms": latency_ms, "ssl_version": ssock.version()}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"status": "failed", "error": str(e), "connection_ms": elapsed_ms}

# ==============================
# WEBSOCKET HANDSHAKE LATENCY
# ==============================
def test_ws_latency(ws_path):
    """Break down WS handshake into TCP connection vs server processing"""
    start = time.perf_counter()
    try:
        key = base64.b64encode(os.urandom(16)).decode("utf-8")
        context = ssl.create_default_context()
        
        # Phase 1: TCP connection + TLS handshake
        tcp_start = time.perf_counter()
        with socket.create_connection((DEEPGRAM_HOST, DEEPGRAM_PORT), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=DEEPGRAM_HOST) as ssock:
                tcp_latency_ms = (time.perf_counter() - tcp_start) * 1000
                
                # Phase 2: Send request
                request_start = time.perf_counter()
                request = (
                    f"GET {ws_path} HTTP/1.1\r\n"
                    f"Host: {DEEPGRAM_HOST}\r\n"
                    f"Authorization: Token {API_KEY}\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\n"
                    f"Sec-WebSocket-Version: 13\r\n\r\n"
                )
                ssock.sendall(request.encode())
                send_latency_ms = (time.perf_counter() - request_start) * 1000
                
                # Phase 3: Receive response (server processing time)
                response_start = time.perf_counter()
                response = b""
                while b"\r\n\r\n" not in response:
                    chunk = ssock.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                server_processing_ms = (time.perf_counter() - response_start) * 1000

                total_latency_ms = (time.perf_counter() - start) * 1000
                response_text = response.decode(errors="ignore")
                
                if "101 Switching Protocols" not in response_text:
                    return {"status": "failed", "error": response_text}

                dg_request_id = None
                for line in response_text.split("\r\n"):
                    if line.lower().startswith("dg-request-id:"):
                        dg_request_id = line.split(":", 1)[1].strip()
                        break

                return {
                    "status": "success",
                    "handshake_ms": total_latency_ms,
                    "tcp_tls_ms": tcp_latency_ms,  # Network path time
                    "send_request_ms": send_latency_ms,  # Usually negligible
                    "server_processing_ms": server_processing_ms,  # Server-side time
                    "dg-request-id": dg_request_id or "none"
                }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"status": "failed", "error": str(e), "handshake_ms": elapsed_ms}

# ==============================
# TCP TRACEROUTE PER ITERATION
# ==============================
TRACEROUTE_TIMEOUT = 30  # seconds per iteration

def run_traceroute(host, port=443, max_hops=30):
    try:
        cmd = ["traceroute", "-T", "-p", str(port), "-m", str(max_hops), host]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TRACEROUTE_TIMEOUT)
        return proc.stdout.strip()
    except Exception as e:
        return f"Error running traceroute: {e}"

def parse_traceroute(output):
    hops_lines = []
    hop_latencies = []
    hop_numbers = []
    rtt_regex = re.compile(r"(\d+\.\d+)\s*ms")
    hop_num_regex = re.compile(r"^\s*(\d+)\s+")
    
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("traceroute"):
            continue
        
        hop_match = hop_num_regex.match(line)
        hop_num = int(hop_match.group(1)) if hop_match else len(hops_lines) + 1
        
        hops_lines.append(line)
        hop_numbers.append(hop_num)
        rtts = [float(m.group(1)) for m in rtt_regex.finditer(line)]
        avg_rtt = sum(rtts)/len(rtts) if rtts else None
        hop_latencies.append(avg_rtt if avg_rtt is not None else "*")

    numeric_latencies = [l for l in hop_latencies if isinstance(l, float)]
    final_hop_latency = numeric_latencies[-1] if numeric_latencies else 0.0
    avg_latency_per_hop = sum(numeric_latencies)/len(numeric_latencies) if numeric_latencies else 0.0
    
    # Analyze hops by position
    if len(numeric_latencies) >= 3:
        # First 3 hops: Often local network/ISP, may indicate client-side network issues
        early_hops = [l for i, l in enumerate(hop_latencies) if i < 3 and isinstance(l, float)]
        early_hops_latency = sum(early_hops) / len(early_hops) if early_hops else 0.0
        
        # Last 3 hops: Often near destination, may indicate server-side/destination network issues
        final_hops = [l for i, l in enumerate(hop_latencies) if i >= len(hop_latencies) - 3 and isinstance(l, float)]
        final_hops_latency = sum(final_hops) / len(final_hops) if final_hops else 0.0
        
        # Middle hops: Typically the network path between client and server
        middle_hops = [l for i, l in enumerate(hop_latencies) if 3 <= i < len(hop_latencies) - 3 and isinstance(l, float)]
        middle_hops_latency = sum(middle_hops) / len(middle_hops) if middle_hops else 0.0
    else:
        early_hops_latency = 0.0
        final_hops_latency = 0.0
        middle_hops_latency = final_hop_latency
    
    return {
        "hops": len(hops_lines),
        "final_hop_latency_ms": final_hop_latency,
        "average_latency_per_hop_ms": avg_latency_per_hop,
        "early_hops_latency_ms": early_hops_latency,  
        "final_hops_latency_ms": final_hops_latency,  
        "middle_hops_latency_ms": middle_hops_latency,  
        "per_hop": [
            {"hop_number": num, "line": line, "avg_latency_ms": lat} 
            for num, line, lat in zip(hop_numbers, hops_lines, hop_latencies)
        ]
    }

# ==============================
# SUMMARY GENERATION FROM FILE
# ==============================
def generate_summary_from_file(model_name=None):
    results_file = f"ws_latency_results_{model_name}.jsonl" if model_name else RESULTS_FILE
    results_path = Path(OUTPUT_DIR) / results_file
    all_results_local = []
    if not results_path.exists():
        print(f"⚠️ Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        for line in f:
            all_results_local.append(json.loads(line.strip()))

    global all_results
    all_results = all_results_local
    generate_summary(model_name)

def generate_summary(model_name=None):
    tcp_latencies = [r["tcp"]["connection_ms"] for r in all_results if r["tcp"]["status"] == "success"]
    ws_latencies = [r["websocket"]["handshake_ms"] for r in all_results if r["websocket"]["status"] == "success"]
    traceroute_hops = [r["traceroute"]["hops"] for r in all_results if "traceroute" in r]
    traceroute_final_latencies = [r["traceroute"]["final_hop_latency_ms"] for r in all_results if "traceroute" in r]
    traceroute_avg_latencies = [r["traceroute"]["average_latency_per_hop_ms"] for r in all_results if "traceroute" in r]

    dns_latencies = [r["dns"]["dns_ms"] for r in all_results if r.get("dns", {}).get("status") == "success" and r["dns"].get("dns_ms") is not None]
    network_path_latencies = [r["latency_breakdown"]["network_path_ms"] for r in all_results if "latency_breakdown" in r and r["latency_breakdown"].get("network_path_ms") is not None]
    server_processing_latencies = [r["latency_breakdown"]["server_processing_ms"] for r in all_results if "latency_breakdown" in r and r["latency_breakdown"].get("server_processing_ms") is not None]
    traceroute_early_hops_latencies = [r["traceroute"]["early_hops_latency_ms"] for r in all_results if "traceroute" in r and r["traceroute"].get("early_hops_latency_ms", 0) > 0]
    traceroute_final_hops_latencies = [r["traceroute"]["final_hops_latency_ms"] for r in all_results if "traceroute" in r and r["traceroute"].get("final_hops_latency_ms", 0) > 0]
    traceroute_middle_hops_latencies = [r["traceroute"]["middle_hops_latency_ms"] for r in all_results if "traceroute" in r and r["traceroute"].get("middle_hops_latency_ms", 0) > 0]

    flagged_iterations = [
        r["iteration"]
        for r in all_results
        if (r["tcp"].get("connection_ms",0) > HIGH_LATENCY_THRESHOLD_MS or
            r["websocket"].get("handshake_ms",0) > HIGH_LATENCY_THRESHOLD_MS)
    ]

    summary = {
        "total_iterations": len(all_results),
        "tcp": {
            "min_ms": min(tcp_latencies) if tcp_latencies else None,
            "max_ms": max(tcp_latencies) if tcp_latencies else None,
            "avg_ms": sum(tcp_latencies)/len(tcp_latencies) if tcp_latencies else None
        },
        "websocket": {
            "min_ms": min(ws_latencies) if ws_latencies else None,
            "max_ms": max(ws_latencies) if ws_latencies else None,
            "avg_ms": sum(ws_latencies)/len(ws_latencies) if ws_latencies else None
        },
        "traceroute": {
            "min_hops": min(traceroute_hops) if traceroute_hops else None,
            "max_hops": max(traceroute_hops) if traceroute_hops else None,
            "avg_hops": sum(traceroute_hops)/len(traceroute_hops) if traceroute_hops else None,
            "min_final_hop_latency_ms": min(traceroute_final_latencies) if traceroute_final_latencies else None,
            "max_final_hop_latency_ms": max(traceroute_final_latencies) if traceroute_final_latencies else None,
            "avg_final_hop_latency_ms": sum(traceroute_final_latencies)/len(traceroute_final_latencies) if traceroute_final_latencies else None,
            "avg_latency_per_hop_ms": sum(traceroute_avg_latencies)/len(traceroute_avg_latencies) if traceroute_avg_latencies else None
        },
        "client_vs_server": {
            "dns": {
                "min_ms": min(dns_latencies) if dns_latencies else None,
                "max_ms": max(dns_latencies) if dns_latencies else None,
                "avg_ms": sum(dns_latencies)/len(dns_latencies) if dns_latencies else None
            },
            "network_path": {
                "min_ms": min(network_path_latencies) if network_path_latencies else None,
                "max_ms": max(network_path_latencies) if network_path_latencies else None,
                "avg_ms": sum(network_path_latencies)/len(network_path_latencies) if network_path_latencies else None
            },
            "server_processing": {
                "min_ms": min(server_processing_latencies) if server_processing_latencies else None,
                "max_ms": max(server_processing_latencies) if server_processing_latencies else None,
                "avg_ms": sum(server_processing_latencies)/len(server_processing_latencies) if server_processing_latencies else None
            },
            "traceroute_early_hops_ms": {
                "min_ms": min(traceroute_early_hops_latencies) if traceroute_early_hops_latencies else None,
                "max_ms": max(traceroute_early_hops_latencies) if traceroute_early_hops_latencies else None,
                "avg_ms": sum(traceroute_early_hops_latencies)/len(traceroute_early_hops_latencies) if traceroute_early_hops_latencies else None,
                "note": "First 3 hops - may indicate client-side/local network issues"
            },
            "traceroute_final_hops_ms": {
                "min_ms": min(traceroute_final_hops_latencies) if traceroute_final_hops_latencies else None,
                "max_ms": max(traceroute_final_hops_latencies) if traceroute_final_hops_latencies else None,
                "avg_ms": sum(traceroute_final_hops_latencies)/len(traceroute_final_hops_latencies) if traceroute_final_hops_latencies else None,
                "note": "Last 3 hops - may indicate server-side/destination network issues"
            },
            "traceroute_middle_hops_ms": {
                "min_ms": min(traceroute_middle_hops_latencies) if traceroute_middle_hops_latencies else None,
                "max_ms": max(traceroute_middle_hops_latencies) if traceroute_middle_hops_latencies else None,
                "avg_ms": sum(traceroute_middle_hops_latencies)/len(traceroute_middle_hops_latencies) if traceroute_middle_hops_latencies else None,
                "note": "Middle hops - typically the network path between client and server"
            }
        },
        "high_latency_iterations": flagged_iterations
    }

    summary_file = f"ws_latency_summary_{model_name}.json" if model_name else SUMMARY_FILE
    summary_path = Path(OUTPUT_DIR) / summary_file
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved in: {OUTPUT_DIR}/{summary_file}")
    if flagged_iterations:
        print(f"⚠️ High latency iterations (> {HIGH_LATENCY_THRESHOLD_MS} ms): {flagged_iterations}")
    if tcp_latencies: print(f"TCP Avg: {summary['tcp']['avg_ms']:.2f} ms, Max: {summary['tcp']['max_ms']:.2f} ms")
    if ws_latencies: print(f"WS Avg: {summary['websocket']['avg_ms']:.2f} ms, Max: {summary['websocket']['max_ms']:.2f} ms")
    if traceroute_final_latencies: print(f"Traceroute Avg final hop latency: {summary['traceroute']['avg_final_hop_latency_ms']:.2f} ms, Avg hops: {summary['traceroute']['avg_hops']:.2f}")
    
    if dns_latencies:
        print(f"\n📊 Client vs Server Breakdown:")
        print(f"  DNS (client-side): Avg {summary['client_vs_server']['dns']['avg_ms']:.2f} ms")
    if network_path_latencies:
        print(f"  Network Path: Avg {summary['client_vs_server']['network_path']['avg_ms']:.2f} ms")
    if server_processing_latencies:
        print(f"  Server Processing: Avg {summary['client_vs_server']['server_processing']['avg_ms']:.2f} ms")
    if traceroute_early_hops_latencies:
        print(f"  Traceroute Early Hops (may indicate client-side): Avg {summary['client_vs_server']['traceroute_early_hops_ms']['avg_ms']:.2f} ms")
    if traceroute_final_hops_latencies:
        print(f"  Traceroute Final Hops (may indicate server-side): Avg {summary['client_vs_server']['traceroute_final_hops_ms']['avg_ms']:.2f} ms")

# ==============================
# RUN TESTS CONTINUOUSLY
# ==============================
def run_tests_continuous(ws_path, model_name):
    global iteration, stop_flag

    if not API_KEY:
        print("❌ DEEPGRAM_API_KEY is not set in the environment. Aborting.")
        return

    # Normalize model name for filename (replace hyphens with underscores)
    model_file_name = model_name.replace("-", "_")
    
    ensure_output_dir()
    log_file = f"ws_test_log_{model_file_name}.txt"
    with open(Path(OUTPUT_DIR) / log_file, "w") as f:
        f.write(f"Deepgram WS Latency Test & Traceroute\nModel: {model_name} ({ws_path})\nStarted: {datetime.now().isoformat()}\n{'='*60}\n")

    try:
        while not stop_flag:
            # Test DNS resolution 
            dns_result = test_dns_resolution()
            
            # Test TCP latency
            tcp_result = test_tcp_latency()
            
            # Test WebSocket with breakdown
            ws_result = test_ws_latency(ws_path)
            
            # Traceroute analysis
            traceroute_output = run_traceroute(DEEPGRAM_HOST, DEEPGRAM_PORT, MAX_TRACEROUTE_HOPS)
            traceroute_metrics = parse_traceroute(traceroute_output)
            
            # Calculate client vs server latency breakdown
            network_latency = ws_result.get("tcp_tls_ms", 0) if ws_result.get("status") == "success" else 0
            server_processing = ws_result.get("server_processing_ms", 0) if ws_result.get("status") == "success" else 0
            
            iteration_result = {
                "iteration": iteration,
                "dns": dns_result,
                "tcp": tcp_result,
                "websocket": ws_result,
                "traceroute": traceroute_metrics,
                "latency_breakdown": {
                    "dns_ms": dns_result.get("dns_ms"),
                    "network_path_ms": network_latency,   # TCP/TLS connection
                    "server_processing_ms": server_processing,  # Server-side
                    "total_handshake_ms": ws_result.get("handshake_ms", 0)
                },
                "timestamp": datetime.now().isoformat()
            }

            append_result_to_file(iteration_result, model_file_name)
            
            # Get IP address(es) for logging
            ip_addresses = dns_result.get('ip_addresses', [])
            ip_str = ', '.join(ip_addresses) if ip_addresses else 'UNKNOWN'
            
            write_log(
                f"Iter {iteration} - Destination IP: {ip_str}",
                model_file_name
            )
            write_log(
                f"Iter {iteration} - DNS: {fmt_ms(dns_result.get('dns_ms', 'FAILED'))} ms, "
                f"TCP: {fmt_ms(tcp_result.get('connection_ms', 'FAILED'))} ms, "
                f"WS: {fmt_ms(ws_result.get('handshake_ms', 'FAILED'))} ms, "
                f"Server: {fmt_ms(server_processing)} ms",
                model_file_name
            )
            write_log(
                f"Iter {iteration} - Traceroute hops: {traceroute_metrics.get('hops')}, "
                f"final hop latency: {fmt_ms(traceroute_metrics.get('final_hop_latency_ms'))} ms",
                model_file_name
            )
            
            # Flag client-side vs server-side issues
            if ws_result.get("status") == "success":
                if (dns_result.get("dns_ms") or 0) > 100:
                    write_log(f"Iter {iteration} - ⚠️ HIGH CLIENT-SIDE DNS: {fmt_ms(dns_result.get('dns_ms'))} ms", model_file_name)
                if network_latency > HIGH_LATENCY_THRESHOLD_MS:
                    write_log(f"Iter {iteration} - ⚠️ HIGH NETWORK PATH LATENCY: {fmt_ms(network_latency)} ms", model_file_name)
                if server_processing > 300:
                    write_log(f"Iter {iteration} - ⚠️ HIGH SERVER PROCESSING: {fmt_ms(server_processing)} ms", model_file_name)

            iteration += 1
            time.sleep(SLEEP_BETWEEN_ITERATIONS)

    finally:
        print(f"\n✅ Test stopped. Total iterations: {iteration-1}")
        generate_summary_from_file(model_file_name)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deepgram WebSocket Latency Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models: {', '.join(sorted(set(MODEL_PATHS.keys())))}
Examples:
  python3 ws_latency_test.py --model=nova-2
  python3 ws_latency_test.py --model=flux
  python3 ws_latency_test.py --model=nova-3
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux",
        choices=list(MODEL_PATHS.keys()),
        help="Deepgram model to test (default: flux)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ws_path = MODEL_PATHS[args.model]
    print(f"🚀 Starting latency test for model: {args.model}")
    print(f"   WebSocket path: {ws_path}\n")
    run_tests_continuous(ws_path, args.model)
