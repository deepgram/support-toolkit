#!/usr/bin/env python3
"""
Stream text to Deepgram TTS websocket and record timing metrics for latency analysis.

Reads a text file with one phrase per line, sends each phrase to Deepgram's TTS
websocket API with a Flush after each phrase, and records detailed timing information
for later analysis.
"""

import os
import sys
import json
import signal
import asyncio
import datetime
import urllib.parse

import click
import websockets


def parse_text_file(filepath: str) -> list[str]:
    """
    Parse a text file and return non-empty, non-comment lines.
    
    Comment lines start with # or //
    """
    phrases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip comment lines
            if line.startswith("#") or line.startswith("//"):
                continue
            phrases.append(line)
    return phrases


def build_websocket_url(
    base_url: str | None,
    model: str,
    sample_rate: int,
    encoding: str = "linear16",
) -> str:
    """Build the websocket URL with query parameters."""
    if base_url:
        # User provided a full URL, use it as-is
        return base_url
    
    # Build URL from components
    params = {
        "model": model,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
    }
    query_string = urllib.parse.urlencode(params)
    return f"wss://api.deepgram.com/v1/speak?{query_string}"


def parse_params_from_url(url: str) -> dict:
    """Extract encoding and sample_rate from URL query params."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    return {
        "model": query.get("model", ["unknown"])[0],
        "encoding": query.get("encoding", ["linear16"])[0],
        "sample_rate": int(query.get("sample_rate", ["24000"])[0]),
    }


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


async def stream_tts(
    input_file: str,
    output_file: str,
    audio_output_file: str,
    url: str | None,
    model: str,
    sample_rate: int,
    verbose: int,
):
    """
    Stream text phrases to Deepgram TTS websocket and record timing data.
    """
    # Parse input file
    phrases = parse_text_file(input_file)
    if not phrases:
        print("No phrases found in input file.", file=sys.stderr)
        return
    
    print(f"Loaded {len(phrases)} phrases from {input_file}", file=sys.stderr)
    
    # Build websocket URL
    ws_url = build_websocket_url(url, model, sample_rate)
    url_params = parse_params_from_url(ws_url)
    
    if verbose >= 1:
        print(f"WebSocket URL: {ws_url}", file=sys.stderr)
    
    # Data structures for tracking
    session_data = {
        "url": ws_url,
        "model": url_params["model"],
        "encoding": url_params["encoding"],
        "sample_rate": url_params["sample_rate"],
        "input_file": input_file,
        "session_start": None,  # Before websocket connection attempt
        "connected_at": None,   # After websocket connection established
        "session_end": None,
    }
    
    # Initialize phrase tracking
    # Each phrase gets: text, sequence_id, text_sent_at, flush_sent_at, packets[]
    phrase_data = []
    for i, text in enumerate(phrases):
        phrase_data.append({
            "sequence_id": i,
            "text": text,
            "text_sent_at": None,
            "flush_sent_at": None,
            "flushed_received_at": None,
            "packets": [],
        })
    
    # Track which phrase we're currently receiving audio for
    current_sequence_id = 0
    
    # Collect all audio bytes
    audio_chunks: list[bytes] = []
    
    # Metadata from server
    server_metadata = None
    
    try:
        session_data["session_start"] = now_iso()
        async with websockets.connect(
            ws_url,
            extra_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
        ) as ws:
            session_data["connected_at"] = now_iso()
            
            # Get request ID from response headers
            request_id = ws.response_headers.get("dg-request-id")
            session_data["request_id"] = request_id
            print(f"Request ID: {request_id}", file=sys.stderr)
            
            # Set up signal handlers for graceful shutdown
            shutdown_event = asyncio.Event()
            
            def handle_signal():
                print("\nReceived interrupt, closing connection...", file=sys.stderr)
                shutdown_event.set()
            
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)
            
            async def sender():
                """Send all phrases with flush after each."""
                nonlocal phrase_data
                
                for i, phrase in enumerate(phrase_data):
                    if shutdown_event.is_set():
                        break
                    
                    text = phrase["text"]
                    
                    # Send Speak message
                    speak_msg = json.dumps({"type": "Speak", "text": text})
                    phrase["text_sent_at"] = now_iso()
                    await ws.send(speak_msg)
                    
                    if verbose >= 2:
                        print(f"Sent phrase {i}: {text[:50]}{'...' if len(text) > 50 else ''}", file=sys.stderr)
                    
                    # Send Flush message
                    flush_msg = json.dumps({"type": "Flush"})
                    phrase["flush_sent_at"] = now_iso()
                    await ws.send(flush_msg)
                    
                    if verbose >= 2:
                        print(f"Sent flush for phrase {i}", file=sys.stderr)
                
                # Send Close message to gracefully end the connection
                if not shutdown_event.is_set():
                    close_msg = json.dumps({"type": "Close"})
                    await ws.send(close_msg)
                    if verbose >= 1:
                        print("Sent Close message", file=sys.stderr)
            
            async def receiver():
                """Receive audio data and control messages."""
                nonlocal current_sequence_id
                nonlocal server_metadata
                nonlocal audio_chunks
                nonlocal phrase_data
                
                async for message in ws:
                    if shutdown_event.is_set():
                        break
                    
                    received_at = now_iso()
                    
                    if isinstance(message, bytes):
                        # Audio data
                        audio_chunks.append(message)
                        
                        # Record packet info for current phrase
                        if current_sequence_id < len(phrase_data):
                            phrase_data[current_sequence_id]["packets"].append({
                                "received_at": received_at,
                                "byte_size": len(message),
                            })
                        
                        if verbose >= 2:
                            print(f"Received audio chunk: {len(message)} bytes for phrase {current_sequence_id}", file=sys.stderr)
                    
                    else:
                        # JSON message
                        try:
                            msg = json.loads(message)
                        except json.JSONDecodeError:
                            print(f"Failed to parse message: {message}", file=sys.stderr)
                            continue
                        
                        msg_type = msg.get("type")
                        
                        if msg_type == "Metadata":
                            server_metadata = msg
                            if verbose >= 1:
                                print(f"Received metadata: model={msg.get('model_name')}", file=sys.stderr)
                        
                        elif msg_type == "Flushed":
                            seq_id = msg.get("sequence_id", current_sequence_id)
                            if seq_id < len(phrase_data):
                                phrase_data[seq_id]["flushed_received_at"] = received_at
                            
                            if verbose >= 1:
                                print(f"Received Flushed for sequence {seq_id}", file=sys.stderr)
                            
                            # Move to next phrase
                            current_sequence_id = seq_id + 1
                        
                        elif msg_type == "Warning":
                            print(f"Warning from server: {msg.get('description')}", file=sys.stderr)
                        
                        elif msg_type == "Error":
                            print(f"Error from server: {msg}", file=sys.stderr)
                        
                        else:
                            if verbose >= 2:
                                print(f"Received message: {msg}", file=sys.stderr)
            
            # Run sender and receiver concurrently
            await asyncio.gather(
                asyncio.create_task(sender()),
                asyncio.create_task(receiver()),
            )
    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"WebSocket connection failed: {e}", file=sys.stderr)
        if hasattr(e, 'headers'):
            print(f"Headers: {e.headers}", file=sys.stderr)
        raise
    
    except websockets.exceptions.ConnectionClosedError as e:
        if verbose >= 1:
            print(f"Connection closed: {e}", file=sys.stderr)
    
    finally:
        session_data["session_end"] = now_iso()
    
    # Build output data
    output_data = {
        "session": session_data,
        "server_metadata": server_metadata,
        "phrases": phrase_data,
    }
    
    # Write JSON output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved timing data to {output_file}", file=sys.stderr)
    
    # Write audio output
    if audio_chunks:
        # For linear16, we need to write a proper WAV header
        audio_data = b"".join(audio_chunks)
        write_wav_file(audio_output_file, audio_data, url_params["sample_rate"], url_params["encoding"])
        print(f"Saved audio to {audio_output_file} ({len(audio_data)} bytes)", file=sys.stderr)
    else:
        print("No audio data received.", file=sys.stderr)


def write_wav_file(filepath: str, audio_data: bytes, sample_rate: int, encoding: str):
    """
    Write audio data to a WAV file with appropriate header.
    
    Supports linear16 (PCM), mulaw, and alaw encodings.
    """
    import struct
    
    channels = 1  # Deepgram TTS is mono
    
    if encoding == "linear16":
        bits_per_sample = 16
        audio_format = 1  # PCM
    elif encoding == "mulaw":
        bits_per_sample = 8
        audio_format = 7  # mu-law
    elif encoding == "alaw":
        bits_per_sample = 8
        audio_format = 6  # A-law
    else:
        # Default to PCM
        bits_per_sample = 16
        audio_format = 1
    
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(audio_data)
    
    # Build WAV header
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)  # File size - 8
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # Format chunk size
    header += struct.pack("<H", audio_format)  # Audio format
    header += struct.pack("<H", channels)  # Channels
    header += struct.pack("<I", sample_rate)  # Sample rate
    header += struct.pack("<I", byte_rate)  # Byte rate
    header += struct.pack("<H", block_align)  # Block align
    header += struct.pack("<H", bits_per_sample)  # Bits per sample
    header += b"data"
    header += struct.pack("<I", data_size)  # Data chunk size
    
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(audio_data)


@click.command(
    help="Stream text to Deepgram TTS websocket and save timing metrics for latency analysis.",
    context_settings={"show_default": True},
)
@click.option(
    "--input", "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input text file with one phrase per line",
)
@click.option(
    "--output", "-o",
    "output_file",
    required=False,
    type=click.Path(),
    help="Output JSON file for timing data (defaults to input filename with .json extension)",
)
@click.option(
    "--audio-output", "-a",
    "audio_output_file",
    required=False,
    type=click.Path(),
    help="Output audio file (defaults to input filename with .wav extension)",
)
@click.option(
    "--url", "-u",
    required=False,
    help="Full Deepgram TTS websocket URL (overrides --model and --sample-rate)",
)
@click.option(
    "--model", "-m",
    default="aura-2-thalia-en",
    help="Deepgram TTS model/voice",
)
@click.option(
    "--sample-rate", "-s",
    default=24000,
    type=int,
    help="Audio sample rate in Hz",
)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Increase verbosity (use -v, -vv, -vvv)",
)
def main(
    input_file: str,
    output_file: str | None,
    audio_output_file: str | None,
    url: str | None,
    model: str,
    sample_rate: int,
    verbose: int,
):
    # Check for API key
    if "DEEPGRAM_API_KEY" not in os.environ:
        print("Error: DEEPGRAM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Generate default output filenames
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    if output_file is None:
        output_file = f"{base_name}.json"
    
    if audio_output_file is None:
        audio_output_file = f"{base_name}.wav"
    
    # Run the async function
    asyncio.run(stream_tts(
        input_file=input_file,
        output_file=output_file,
        audio_output_file=audio_output_file,
        url=url,
        model=model,
        sample_rate=sample_rate,
        verbose=verbose,
    ))


if __name__ == "__main__":
    main()
