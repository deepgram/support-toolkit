#!/usr/bin/env python3
"""
Analyze latency metrics from Deepgram TTS streaming timing data.

Calculates:
- Time To First Byte (TTFB) - session level only
- TTFB with Network Latency (TTFB + estimated RTT) - session level only
- Time To Last Byte (TTLB) - session level only
- Real-Time Factor (RTF) per packet and per phrase
- Jitter (inter-arrival time variability)

Input: JSON file produced by stream_tts.py
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import TextIO

import click


def parse_iso_timestamp(iso_str: str) -> datetime:
    """Parse an ISO format timestamp string to datetime."""
    # Handle both with and without timezone
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    return datetime.fromisoformat(iso_str)


def timestamp_diff_seconds(t1: str, t2: str) -> float:
    """Calculate difference in seconds between two ISO timestamps (t2 - t1)."""
    dt1 = parse_iso_timestamp(t1)
    dt2 = parse_iso_timestamp(t2)
    return (dt2 - dt1).total_seconds()


def calculate_audio_duration_seconds(byte_size: int, sample_rate: int, encoding: str) -> float:
    """
    Calculate audio duration from byte size based on encoding.
    
    For linear16: 2 bytes per sample, mono
    For mulaw/alaw: 1 byte per sample, mono
    """
    if encoding == "linear16":
        bytes_per_sample = 2
    elif encoding in ("mulaw", "alaw"):
        bytes_per_sample = 1
    else:
        # Default to linear16
        bytes_per_sample = 2
    
    channels = 1  # Deepgram TTS is mono
    num_samples = byte_size / (bytes_per_sample * channels)
    return num_samples / sample_rate


@dataclass
class PacketMetrics:
    """Metrics for a single audio packet."""
    packet_index: int
    received_at: str
    byte_size: int
    audio_duration_seconds: float
    
    # Inter-arrival metrics (None for first packet in phrase)
    inter_arrival_seconds: float | None = None
    
    # Cumulative metrics (within phrase)
    cumulative_audio_seconds: float = 0.0  # Total audio received up to and including this packet
    wall_clock_since_first_packet: float = 0.0  # Time since first packet in phrase
    cumulative_rtf: float | None = None  # cumulative_audio / wall_clock (None for first packet)


@dataclass
class PhraseMetrics:
    """Metrics for a single phrase."""
    sequence_id: int
    text: str
    text_sent_at: str
    flush_sent_at: str
    flushed_received_at: str | None
    
    # Audio metrics
    total_audio_duration_seconds: float = 0.0
    total_bytes: int = 0
    packet_count: int = 0
    
    # Delivery metrics
    first_packet_at: str | None = None
    last_packet_at: str | None = None
    delivery_duration_seconds: float | None = None  # Time from first to last packet
    
    # Per-packet data
    packets: list[PacketMetrics] = field(default_factory=list)
    
    # Overall RTF for this phrase: audio_duration / delivery_duration
    rtf: float | None = None
    
    # Min cumulative RTF within this phrase (did playback ever fall behind?)
    min_cumulative_rtf: float | None = None
    
    # Jitter within this phrase
    jitter_seconds: float | None = None  # Std dev of inter-arrival times


@dataclass 
class SessionMetrics:
    """Aggregate metrics across all phrases in a session."""
    # Overall timing
    session_start: str      # Before websocket connection attempt
    connected_at: str       # After websocket connection established
    session_end: str
    session_duration_seconds: float
    
    # Phrase-level aggregates
    num_phrases: int
    total_packets: int
    total_audio_duration_seconds: float
    total_bytes: int
    
    # TTFB: time from first text sent to first audio byte received (single value)
    ttfb_seconds: float | None = None
    
    # TTFB with network latency: time from session start (before websocket connection)
    # to first audio byte received - captures full end-to-end latency
    ttfb_with_network_latency_seconds: float | None = None
    
    # TTLB: time from first text sent to last audio byte received (single value)
    ttlb_seconds: float | None = None
    
    # Overall RTF: total_audio / delivery_time
    overall_rtf: float | None = None
    
    # Min cumulative RTF across entire session (did stream ever fall behind?)
    min_cumulative_rtf: float | None = None
    
    # Jitter (across all inter-arrival times in session)
    jitter_seconds: float | None = None  # Std dev of all inter-arrival times
    mean_inter_arrival_seconds: float | None = None


def analyze_phrase(phrase_data: dict, sample_rate: int, encoding: str) -> PhraseMetrics:
    """Analyze a single phrase and compute its metrics."""
    metrics = PhraseMetrics(
        sequence_id=phrase_data["sequence_id"],
        text=phrase_data["text"],
        text_sent_at=phrase_data["text_sent_at"],
        flush_sent_at=phrase_data["flush_sent_at"],
        flushed_received_at=phrase_data.get("flushed_received_at"),
    )
    
    packets_data = phrase_data.get("packets", [])
    metrics.packet_count = len(packets_data)
    
    if not packets_data:
        return metrics
    
    # Process each packet
    first_received_at: str | None = None
    prev_received_at: str | None = None
    cumulative_audio = 0.0
    cumulative_rtf_values = []
    
    for i, pkt in enumerate(packets_data):
        byte_size = pkt["byte_size"]
        received_at = pkt["received_at"]
        audio_dur = calculate_audio_duration_seconds(byte_size, sample_rate, encoding)
        
        # Track cumulative audio
        cumulative_audio += audio_dur
        
        packet_metrics = PacketMetrics(
            packet_index=i,
            received_at=received_at,
            byte_size=byte_size,
            audio_duration_seconds=audio_dur,
            cumulative_audio_seconds=cumulative_audio,
        )
        
        # Track first packet time
        if first_received_at is None:
            first_received_at = received_at
        
        # Calculate inter-arrival time for packets after the first
        if prev_received_at is not None:
            inter_arrival = timestamp_diff_seconds(prev_received_at, received_at)
            packet_metrics.inter_arrival_seconds = inter_arrival
        
        # Calculate cumulative RTF (wall clock since first packet vs cumulative audio)
        # For first packet, wall_clock is 0, so we skip cumulative RTF
        if i > 0 and first_received_at is not None:
            wall_clock = timestamp_diff_seconds(first_received_at, received_at)
            packet_metrics.wall_clock_since_first_packet = wall_clock
            if wall_clock > 0:
                packet_metrics.cumulative_rtf = cumulative_audio / wall_clock
                cumulative_rtf_values.append(packet_metrics.cumulative_rtf)
        
        metrics.packets.append(packet_metrics)
        metrics.total_audio_duration_seconds += audio_dur
        metrics.total_bytes += byte_size
        
        prev_received_at = received_at
    
    # Record first and last packet times
    metrics.first_packet_at = metrics.packets[0].received_at
    metrics.last_packet_at = metrics.packets[-1].received_at
    
    # Calculate delivery duration: time from first to last packet
    metrics.delivery_duration_seconds = timestamp_diff_seconds(
        metrics.first_packet_at,
        metrics.last_packet_at
    )
    
    # Calculate phrase RTF: audio_duration / delivery_duration
    if metrics.delivery_duration_seconds and metrics.delivery_duration_seconds > 0:
        metrics.rtf = metrics.total_audio_duration_seconds / metrics.delivery_duration_seconds
    elif metrics.packet_count == 1:
        # Single packet - can't compute RTF this way
        metrics.rtf = None
    
    # Min cumulative RTF (did this phrase ever fall behind?)
    if cumulative_rtf_values:
        metrics.min_cumulative_rtf = min(cumulative_rtf_values)
    
    # Calculate jitter (std dev of inter-arrival times) for this phrase
    inter_arrival_times = [
        p.inter_arrival_seconds 
        for p in metrics.packets 
        if p.inter_arrival_seconds is not None
    ]
    if len(inter_arrival_times) >= 2:
        metrics.jitter_seconds = statistics.stdev(inter_arrival_times)
    elif len(inter_arrival_times) == 1:
        metrics.jitter_seconds = 0.0
    
    return metrics


def analyze_session(data: dict) -> tuple[SessionMetrics, list[PhraseMetrics]]:
    """Analyze all phrases in a session and compute aggregate metrics."""
    session = data["session"]
    sample_rate = session["sample_rate"]
    encoding = session["encoding"]
    
    # Analyze each phrase
    phrase_metrics_list: list[PhraseMetrics] = []
    for phrase_data in data["phrases"]:
        pm = analyze_phrase(phrase_data, sample_rate, encoding)
        phrase_metrics_list.append(pm)
    
    # Compute session-level metrics
    session_duration = timestamp_diff_seconds(session["session_start"], session["session_end"])
    
    session_metrics = SessionMetrics(
        session_start=session["session_start"],
        connected_at=session["connected_at"],
        session_end=session["session_end"],
        session_duration_seconds=session_duration,
        num_phrases=len(phrase_metrics_list),
        total_packets=sum(pm.packet_count for pm in phrase_metrics_list),
        total_audio_duration_seconds=sum(pm.total_audio_duration_seconds for pm in phrase_metrics_list),
        total_bytes=sum(pm.total_bytes for pm in phrase_metrics_list),
    )
    
    # Find first text sent and first/last audio received across all phrases
    phrases_with_packets = [pm for pm in phrase_metrics_list if pm.packets]
    
    if phrases_with_packets:
        # First text sent (should be phrase 0)
        first_text_sent = phrase_metrics_list[0].text_sent_at
        
        # First audio received (earliest first_packet_at across all phrases)
        first_audio_received = min(pm.first_packet_at for pm in phrases_with_packets)
        
        # Last audio received (latest last_packet_at across all phrases)
        last_audio_received = max(pm.last_packet_at for pm in phrases_with_packets)
        
        # TTFB: time from first text sent to first audio byte
        session_metrics.ttfb_seconds = timestamp_diff_seconds(first_text_sent, first_audio_received)
        
        # TTFB with network latency: time from session start (before websocket opens)
        # to first audio byte received
        session_metrics.ttfb_with_network_latency_seconds = timestamp_diff_seconds(
            session_metrics.session_start, first_audio_received
        )
        
        # TTLB: time from first text sent to last audio byte
        session_metrics.ttlb_seconds = timestamp_diff_seconds(first_text_sent, last_audio_received)
        
        # Overall RTF: total audio / delivery time (first packet to last packet)
        delivery_time = timestamp_diff_seconds(first_audio_received, last_audio_received)
        if delivery_time > 0:
            session_metrics.overall_rtf = session_metrics.total_audio_duration_seconds / delivery_time
        
        # Calculate session-wide cumulative RTF
        # Track cumulative audio vs wall clock since first packet across ALL packets in order
        all_packets_with_times: list[tuple[str, float]] = []  # (received_at, audio_duration)
        for pm in phrase_metrics_list:
            for pkt in pm.packets:
                all_packets_with_times.append((pkt.received_at, pkt.audio_duration_seconds))
        
        # Sort by received_at to ensure correct order
        all_packets_with_times.sort(key=lambda x: parse_iso_timestamp(x[0]))
        
        if len(all_packets_with_times) > 1:
            first_packet_time = all_packets_with_times[0][0]
            cumulative_audio = all_packets_with_times[0][1]  # First packet's audio
            cumulative_rtf_values = []
            
            for received_at, audio_dur in all_packets_with_times[1:]:
                cumulative_audio += audio_dur
                wall_clock = timestamp_diff_seconds(first_packet_time, received_at)
                if wall_clock > 0:
                    cumulative_rtf_values.append(cumulative_audio / wall_clock)
            
            if cumulative_rtf_values:
                session_metrics.min_cumulative_rtf = min(cumulative_rtf_values)
    
    # Aggregate jitter (std dev of all inter-arrival times across session)
    all_inter_arrival_times = []
    for pm in phrase_metrics_list:
        for pkt in pm.packets:
            if pkt.inter_arrival_seconds is not None:
                all_inter_arrival_times.append(pkt.inter_arrival_seconds)
    
    if all_inter_arrival_times:
        session_metrics.mean_inter_arrival_seconds = statistics.mean(all_inter_arrival_times)
        if len(all_inter_arrival_times) >= 2:
            session_metrics.jitter_seconds = statistics.stdev(all_inter_arrival_times)
        else:
            session_metrics.jitter_seconds = 0.0
    
    return session_metrics, phrase_metrics_list


def format_duration_ms(seconds: float | None) -> str:
    """Format duration in milliseconds."""
    if seconds is None:
        return "N/A"
    return f"{seconds * 1000:.2f} ms"


def format_rtf(rtf: float | None) -> str:
    """Format RTF value."""
    if rtf is None:
        return "N/A"
    return f"{rtf:.2f}x"


def print_report(
    session_metrics: SessionMetrics, 
    phrase_metrics_list: list[PhraseMetrics],
    verbose: int,
    output: TextIO,
):
    """Print a human-readable report of the metrics."""
    print("=" * 70, file=output)
    print("DEEPGRAM TTS LATENCY ANALYSIS REPORT", file=output)
    print("=" * 70, file=output)
    print(file=output)
    
    # Session overview
    print("SESSION OVERVIEW", file=output)
    print("-" * 40, file=output)
    print(f"  Duration:        {format_duration_ms(session_metrics.session_duration_seconds)}", file=output)
    print(f"  Phrases:         {session_metrics.num_phrases}", file=output)
    print(f"  Total packets:   {session_metrics.total_packets}", file=output)
    print(f"  Total audio:     {format_duration_ms(session_metrics.total_audio_duration_seconds)}", file=output)
    print(f"  Total bytes:     {session_metrics.total_bytes:,}", file=output)
    print(file=output)
    
    # Latency metrics
    print("LATENCY", file=output)
    print("-" * 40, file=output)
    print(f"  TTFB:            {format_duration_ms(session_metrics.ttfb_seconds)}", file=output)
    print(f"  TTFB (incl net): {format_duration_ms(session_metrics.ttfb_with_network_latency_seconds)}", file=output)
    print(f"  TTLB:            {format_duration_ms(session_metrics.ttlb_seconds)}", file=output)
    print(f"  Overall RTF:     {format_rtf(session_metrics.overall_rtf)}", file=output)
    print(file=output)
    
    # Streaming health - the critical "did we fall behind" metric
    print("STREAMING HEALTH", file=output)
    print("-" * 40, file=output)
    print("  (Min cumulative RTF >= 1.0 means stream never fell behind real-time)", file=output)
    print(f"  Min cumulative RTF: {format_rtf(session_metrics.min_cumulative_rtf)}", file=output)
    if session_metrics.min_cumulative_rtf is not None:
        if session_metrics.min_cumulative_rtf >= 1.0:
            print("  Status:          ✓ Stream kept ahead of real-time", file=output)
        else:
            print("  Status:          ✗ Stream fell behind real-time", file=output)
    print(file=output)
    
    # Jitter summary
    print("JITTER (Inter-Arrival Time Variability)", file=output)
    print("-" * 40, file=output)
    print(f"  Mean IAT:        {format_duration_ms(session_metrics.mean_inter_arrival_seconds)}", file=output)
    print(f"  Jitter (σ):      {format_duration_ms(session_metrics.jitter_seconds)}", file=output)
    print(file=output)
    
    # Per-phrase details (if verbose)
    if verbose >= 1:
        print("=" * 70, file=output)
        print("PER-PHRASE METRICS", file=output)
        print("=" * 70, file=output)
        
        for pm in phrase_metrics_list:
            print(file=output)
            text_preview = pm.text[:50] + "..." if len(pm.text) > 50 else pm.text
            print(f"Phrase {pm.sequence_id}: \"{text_preview}\"", file=output)
            print("-" * 40, file=output)
            print(f"  Audio duration:  {format_duration_ms(pm.total_audio_duration_seconds)}", file=output)
            print(f"  Delivery time:   {format_duration_ms(pm.delivery_duration_seconds)}", file=output)
            print(f"  Packets:         {pm.packet_count}", file=output)
            print(f"  Bytes:           {pm.total_bytes:,}", file=output)
            print(f"  RTF:             {format_rtf(pm.rtf)}", file=output)
            print(f"  Min cum. RTF:    {format_rtf(pm.min_cumulative_rtf)}", file=output)
            print(f"  Jitter (σ):      {format_duration_ms(pm.jitter_seconds)}", file=output)
            
            # Per-packet details (if very verbose)
            if verbose >= 2 and pm.packets:
                print(file=output)
                print("  Packets:", file=output)
                print(f"    {'#':>3}  {'Bytes':>8}  {'Audio':>10}  {'IAT':>10}  {'Cum.RTF':>10}", file=output)
                for pkt in pm.packets:
                    iat_str = format_duration_ms(pkt.inter_arrival_seconds) if pkt.inter_arrival_seconds is not None else "-"
                    cum_rtf_str = format_rtf(pkt.cumulative_rtf) if pkt.cumulative_rtf is not None else "-"
                    print(
                        f"    {pkt.packet_index:>3}  "
                        f"{pkt.byte_size:>8}  "
                        f"{format_duration_ms(pkt.audio_duration_seconds):>10}  "
                        f"{iat_str:>10}  "
                        f"{cum_rtf_str:>10}",
                        file=output
                    )


def metrics_to_dict(
    session_metrics: SessionMetrics,
    phrase_metrics_list: list[PhraseMetrics],
) -> dict:
    """Convert metrics to a JSON-serializable dictionary."""
    
    def packet_to_dict(pkt: PacketMetrics) -> dict:
        return {
            "packet_index": pkt.packet_index,
            "received_at": pkt.received_at,
            "byte_size": pkt.byte_size,
            "audio_duration_seconds": pkt.audio_duration_seconds,
            "inter_arrival_seconds": pkt.inter_arrival_seconds,
            "cumulative_audio_seconds": pkt.cumulative_audio_seconds,
            "wall_clock_since_first_packet": pkt.wall_clock_since_first_packet,
            "cumulative_rtf": pkt.cumulative_rtf,
        }
    
    def phrase_to_dict(pm: PhraseMetrics) -> dict:
        return {
            "sequence_id": pm.sequence_id,
            "text": pm.text,
            "text_sent_at": pm.text_sent_at,
            "flush_sent_at": pm.flush_sent_at,
            "flushed_received_at": pm.flushed_received_at,
            "total_audio_duration_seconds": pm.total_audio_duration_seconds,
            "total_bytes": pm.total_bytes,
            "packet_count": pm.packet_count,
            "first_packet_at": pm.first_packet_at,
            "last_packet_at": pm.last_packet_at,
            "delivery_duration_seconds": pm.delivery_duration_seconds,
            "rtf": pm.rtf,
            "min_cumulative_rtf": pm.min_cumulative_rtf,
            "jitter_seconds": pm.jitter_seconds,
            "packets": [packet_to_dict(pkt) for pkt in pm.packets],
        }
    
    return {
        "session": {
            "session_start": session_metrics.session_start,
            "connected_at": session_metrics.connected_at,
            "session_end": session_metrics.session_end,
            "session_duration_seconds": session_metrics.session_duration_seconds,
            "ttfb_seconds": session_metrics.ttfb_seconds,
            "ttfb_with_network_latency_seconds": session_metrics.ttfb_with_network_latency_seconds,
            "ttlb_seconds": session_metrics.ttlb_seconds,
            "overall_rtf": session_metrics.overall_rtf,
            "min_cumulative_rtf": session_metrics.min_cumulative_rtf,
            "num_phrases": session_metrics.num_phrases,
            "total_packets": session_metrics.total_packets,
            "total_audio_duration_seconds": session_metrics.total_audio_duration_seconds,
            "total_bytes": session_metrics.total_bytes,
            "jitter": {
                "mean_inter_arrival_seconds": session_metrics.mean_inter_arrival_seconds,
                "stdev_seconds": session_metrics.jitter_seconds,
            },
        },
        "phrases": [phrase_to_dict(pm) for pm in phrase_metrics_list],
    }


@click.command(
    help="Analyze latency metrics from Deepgram TTS timing data.",
    context_settings={"show_default": True},
)
@click.option(
    "--input", "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON file from stream_tts.py",
)
@click.option(
    "--output", "-o",
    "output_file",
    required=False,
    type=click.Path(),
    help="Output JSON file for computed metrics (optional)",
)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Increase verbosity: -v for per-phrase, -vv for per-packet details",
)
@click.option(
    "--json-only",
    is_flag=True,
    default=False,
    help="Output only JSON, no human-readable report",
)
def main(
    input_file: str,
    output_file: str | None,
    verbose: int,
    json_only: bool,
):
    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Analyze
    session_metrics, phrase_metrics_list = analyze_session(data)
    
    # Print human-readable report (unless json-only)
    if not json_only:
        import sys
        print_report(session_metrics, phrase_metrics_list, verbose, sys.stdout)
    
    # Output JSON if requested
    if output_file or json_only:
        metrics_dict = metrics_to_dict(session_metrics, phrase_metrics_list)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=2)
            if not json_only:
                print(f"\nMetrics saved to {output_file}")
        else:
            # json-only mode with no output file: print to stdout
            print(json.dumps(metrics_dict, indent=2))


if __name__ == "__main__":
    main()
