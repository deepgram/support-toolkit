#!/home/jjmaldonis/miniconda3/envs/deepgram/bin/python
from typing import AsyncGenerator, BinaryIO, Callable
import os
import io
import sys
import json
import struct
import signal
import asyncio
import datetime
import warnings
import threading
import traceback
import urllib.parse
import concurrent.futures

import wave
import click
import pyaudio
import librosa
import websockets
import asyncstdlib
import click.shell_completion
from tqdm.asyncio import tqdm
from pydub import AudioSegment  # type: ignore


def format_time(seconds: float) -> str:
    if seconds < 0:
        return f"-{format_time(-seconds)}"
    hours = seconds // 3600
    seconds = seconds - (hours * 3600)
    minutes = seconds // 60
    seconds = seconds - (minutes * 60)
    milliseconds = (seconds - int(seconds)) * 1000
    # Make sure the milliseconds string is 2 digits
    milliseconds_str = f"{int(milliseconds / 10)}"
    if len(milliseconds_str) == 1:
        milliseconds_str += "0"
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds_str}"


# stderr capturer context manager
class CapturedStderr:
    def __init__(self):
        self.stderr = io.StringIO()
        self._stderr = sys.stderr

    def __enter__(self):
        sys.stderr = self.stderr
        return self.stderr

    def __exit__(self, *args):
        sys.stderr = self._stderr

    def __str__(self):
        return self.stderr.getvalue()

    def print(self):
        print(self.stderr.getvalue())


class Microphone:
    """A simplified async microphone class that yields audio chunks."""

    def __init__(self, output_filename: str | None = None) -> None:
        self.output_filename: None | str = output_filename
        self._wav_file: wave.Wave_write | None = None
        self._audio: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        )  # Increased to 2
        self._stop_event = threading.Event()
        self._audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(
            maxsize=100
        )  # Changed to asyncio.Queue
        self._audio_thread: threading.Thread | None = None

        # Default audio parameters
        self._format = pyaudio.paInt16
        self._channels = 1  # Mono
        self._rate = 16000  # 16kHz
        self._chunk = 1024  # Buffer size

        try:
            self._audio = pyaudio.PyAudio()
        except Exception as e:
            self._executor.shutdown(wait=False)
            raise RuntimeError(f"Failed to initialize PyAudio: {e}") from e

        if self.output_filename is not None:
            self._wav_file = wave.open(self.output_filename, "wb")
            self._wav_file.setnchannels(self._channels)
            self._wav_file.setsampwidth(self._audio.get_sample_size(self._format))
            self._wav_file.setframerate(self._rate)
            print(f"Recording audio to {self.output_filename}", file=sys.stderr)

    async def _audio_capture_loop(self) -> None:
        """Async loop that continuously captures audio."""
        try:
            while (
                not self._stop_event.is_set()
                and self._stream
                and self._stream.is_active()
            ):
                try:
                    # Run the blocking read in the executor
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        self._stream.read,
                        self._chunk,
                        False,  # exception_on_overflow=False
                    )

                    if self._wav_file:
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor, self._wav_file.writeframes, chunk
                        )

                    try:
                        await asyncio.wait_for(
                            self._audio_queue.put(chunk), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        # If queue is full, drop the oldest chunk
                        try:
                            self._audio_queue.get_nowait()
                            await self._audio_queue.put(chunk)
                        except asyncio.QueueEmpty:
                            pass

                except Exception as e:
                    if not self._stop_event.is_set():
                        await self._audio_queue.put(e)
                        break

                # Add a small yield to allow other coroutines to run
                await asyncio.sleep(0)

        finally:
            try:
                await asyncio.wait_for(self._audio_queue.put(None), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """
        Creates an async generator that yields chunks of audio data.
        Yields:
            bytes: Chunks of audio data
        """
        if self._stream is not None:
            raise RuntimeError("Stream already active")
        if self._audio is None:
            raise RuntimeError("Stream was closed")

        self._stream = self._audio.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            output=False,
            frames_per_buffer=self._chunk,
        )

        # Start the audio capture loop as a task
        capture_task = asyncio.create_task(self._audio_capture_loop())

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)

                    if chunk is None:  # End of stream
                        break
                    if isinstance(chunk, Exception):  # Error occurred in capture
                        raise chunk

                    yield chunk

                except asyncio.TimeoutError:
                    if self._stop_event.is_set():
                        break
                    continue

        except asyncio.CancelledError:
            self._stop_event.set()
            raise
        except Exception as e:
            self._stop_event.set()
            raise RuntimeError(f"Error in audio stream: {e}") from e
        finally:
            self._stop_event.set()
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass

            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None
            await self.close()

    async def close(self) -> None:
        """Closes the audio stream and cleans up resources."""
        self._stop_event.set()

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

        self._executor.shutdown(wait=True)


def generate_audio_header(
    encoding: str,
    sample_rate: int,
    channels: int,
    datasize: int | None = None,
):
    """
    Generate an audio header for WAV or mu-law encoded audio.

    Args:
        encoding (str): Audio encoding type ('wav' or 'mulaw')
        sample_rate (int): Sample rate in Hz
        channels (int): Number of audio channels
        datasize (int | None): Size of audio data in bytes (None for streaming)

    Returns:
        bytes: Generated audio header

    Raises:
        ValueError: If encoding is not 'wav' or 'mulaw'
    """
    if encoding.lower() not in ["wav", "linear16", "mulaw"]:
        raise ValueError("Encoding must be either 'wav' or 'mulaw'")

    # Set format-specific parameters
    if encoding.lower() in ("wav", "linear16"):
        format_code = 1  # PCM
        bits_per_sample = 16
        block_align = channels * (bits_per_sample // 8)
    else:  # mulaw
        format_code = 7  # mu-law
        bits_per_sample = 8
        block_align = channels  # 1 byte per channel for mu-law

    byte_rate = sample_rate * channels * (bits_per_sample // 8)

    # RIFF chunk
    data = b"RIFF"
    if datasize is not None:
        filesize = 44 - 8 + datasize  # Total size - 8 bytes for RIFF header
    else:
        filesize = 0
    data += struct.pack("<I", filesize)
    data += b"WAVE"

    # Format chunk
    data += b"fmt "
    data += struct.pack("<I", 16)  # Format chunk size
    data += struct.pack("<H", format_code)  # Audio format
    data += struct.pack("<H", channels)  # Number of channels
    data += struct.pack("<I", sample_rate)  # Sample rate
    data += struct.pack("<I", byte_rate)  # Byte rate
    data += struct.pack("<H", block_align)  # Block align
    data += struct.pack("<H", bits_per_sample)  # Bits per sample

    # Data chunk
    data += b"data"
    data += struct.pack(
        "<I", datasize if datasize is not None else 0
    )  # Data chunk size

    return data


def find_data_chunk_start_in_wav_file(filename: str):
    with open(filename, "rb") as f:
        # Read RIFF header
        riff_header = f.read(12)
        if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
            raise ValueError("Not a valid WAV file")

        # Walk through chunks to find 'data'
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                raise ValueError("Data chunk not found")

            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:8])[0]

            if chunk_id == b"data":
                # Found the data chunk! Current position is start of data
                return f.tell()
            else:
                # Skip this chunk and continue looking
                f.seek(chunk_size, 1)  # Skip chunk_size bytes from current position
                # Handle odd-sized chunks (WAV spec requires even alignment)
                if chunk_size % 2:
                    f.seek(1, 1)


class VerifyAudioFile:
    @classmethod
    def _validate_results_are_close_to_initial_data(
        cls,
        initial_data: dict[str, int | float | None],
        channels: int,
        sample_rate: int,
        duration: float,
    ):
        if (
            initial_data["channels"] is not None
            and initial_data["channels"] != channels
        ):
            raise ValueError(
                f"Expected {initial_data['channels']} channels, got {channels}."
            )
        if (
            initial_data["sample_rate"] is not None
            and initial_data["sample_rate"] != sample_rate
        ):
            raise ValueError(
                f"Expected {initial_data['sample_rate']} sample rate, got {sample_rate}."
            )
        if (
            initial_data["duration"] is not None
            and initial_data["duration"] != duration
        ):
            raise ValueError(
                f"Expected {initial_data['duration']} duration, got {duration}."
            )

    @classmethod
    def _load_with_wave(
        cls, filename: str, initial_data: dict[str, int | float | None]
    ) -> tuple[int, int, int, float] | None:
        try:
            with wave.open(filename, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()

                # Normally we'd use `wav_file.getnframes()` to get the number of frames, but the header data is frequently unreliable.
                # Instead, we calculate the number of frames based on the file size.
                data_start = find_data_chunk_start_in_wav_file(filename)
                file_size = os.path.getsize(filename)
                data_size = file_size - data_start
                frames = data_size // (channels * sample_width)

                duration = frames / float(sample_rate)
            cls._validate_results_are_close_to_initial_data(
                initial_data, channels, sample_rate, duration
            )
            return channels, sample_width, sample_rate, duration
        except Exception as e:
            warnings.warn(str(e), stacklevel=1)
            try:
                del channels, sample_rate, duration
            except UnboundLocalError:
                pass
            return None

    @classmethod
    def _load_with_pydub(
        cls, filename: str, initial_data: dict[str, int | float | None]
    ) -> tuple[int, int, int, float] | None:
        try:
            audio = AudioSegment.from_file(filename)
            channels = audio.channels
            sample_width = audio.sample_width
            sample_rate = audio.frame_rate
            duration = len(audio) / 1000.0  # pydub uses milliseconds
            cls._validate_results_are_close_to_initial_data(
                initial_data, channels, sample_rate, duration  # type: ignore
            )
            return channels, sample_width, sample_rate, duration
        except Exception as e:
            warnings.warn(str(e), stacklevel=1)
            try:
                del channels, sample_rate, duration
            except UnboundLocalError:
                pass
            return None

    @classmethod
    def _load_raw_pcm(
        cls,
        filename: str,
        initial_data: dict[str, int | float | None],
    ) -> tuple[int, int, int, float] | None:
        try:
            with open(filename, "rb") as f:
                data = f.read()
            if initial_data["channels"] is None or initial_data["sample_rate"] is None:
                raise ValueError(
                    "Channels and sample rate must be provided for raw audio."
                )
            channels = int(initial_data["channels"])  # type: ignore
            sample_rate = int(initial_data["sample_rate"])  # type: ignore
            header = generate_audio_header("linear16", sample_rate, channels, len(data))
            audio = io.BytesIO(header + data)
            # Now use the `wave` module to read the audio
            with wave.open(audio, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / float(sample_rate)
            cls._validate_results_are_close_to_initial_data(
                initial_data, channels, sample_rate, duration
            )
            return channels, sample_width, sample_rate, duration
        except Exception as e:
            warnings.warn(str(e), stacklevel=1)
            try:
                del channels, sample_rate, duration
            except UnboundLocalError:
                pass
            return None

    @classmethod
    def _load_mulaw_with_librosa(
        cls,
        filename: str | BinaryIO,
        initial_data: dict[str, int | float | None],
    ) -> tuple[int, int, int, float] | None:
        try:
            if initial_data["sample_rate"] is None:
                raise ValueError("Sample rate must be provided for mu-law audio.")
            sample_rate = int(initial_data["sample_rate"])  # type: ignore
            y, sr = librosa.load(filename, sr=sample_rate, mono=True)
            duration = len(y) / sr
            expected_samples = duration * sample_rate
            if isinstance(filename, str):
                file_size = os.path.getsize(filename)
            elif isinstance(filename, io.BytesIO):
                file_size = len(filename.getbuffer())
            else:
                raise ValueError("Unsupported filename type.")
            sample_width = round(file_size / expected_samples)
            # Determine number of channels
            if y.ndim == 1:
                channels = 1
            else:
                channels = y.shape[0]  # First dimension is channels

            cls._validate_results_are_close_to_initial_data(
                initial_data, channels, sample_rate, duration  # type: ignore
            )
            return channels, sample_width, sample_rate, duration
        except Exception as e:
            warnings.warn(str(e), stacklevel=1)
            try:
                del channels, sample_rate, duration
            except UnboundLocalError:
                pass
            return None

    @classmethod
    def analyze(
        cls,
        filename,
        encoding: str | None = None,
        channels: int | None = None,
        sample_rate: int | None = None,
        duration: float | None = None,
    ):
        """Analyze the audio file and return its properties."""
        initial_data = {
            "channels": channels,
            "sample_rate": sample_rate,
            "duration": duration,
        }
        # Remove from local scope because they are set below
        del channels, sample_rate, duration

        # Capture stderr for this function
        with CapturedStderr() as stderr:
            # Try to open the file with wave first
            result = cls._load_with_wave(filename, initial_data)
            if result is not None:
                return result

            # If that fails, try to open the file with pydub
            result = cls._load_with_pydub(filename, initial_data)
            if result is not None:
                return result

            # If that fails, try raw PCM audio
            if encoding == "linear16":
                result = cls._load_raw_pcm(filename, initial_data)
                if result is not None:
                    return result

            if encoding == "mulaw":
                # librosa does a better job handling mu-law audio than wave or pydub, particularly for mulaw files with a sample width greater than 1
                # we must assume the input sample rate is correct
                result = cls._load_mulaw_with_librosa(filename, initial_data)
                if result is not None:
                    return result

                # As a last attempt, try adding a MU-law WAV header and re-reading with librosa
                new_mulaw_header = generate_audio_header(
                    "mulaw",
                    initial_data["sample_rate"],  # type: ignore
                    initial_data["channels"],  # type: ignore
                    os.path.getsize(filename),
                )
                with open(filename, "rb") as f:
                    raw_data = f.read()
                audio = io.BytesIO(new_mulaw_header + raw_data)
                result = cls._load_mulaw_with_librosa(audio, initial_data)
                if result is not None:
                    return result

            stderr_output = stderr.read()
            if stderr_output.strip():
                raise ValueError(
                    f"Could not analyze audio file {filename}\n:\n{stderr_output}"
                )

        raise ValueError(
            f"Could not verify metadata of audio file `{filename}`. Received {initial_data}."
        )


def calculate_chunk_parameters(
    channels, sample_width, sample_rate, preferred_duration=0.1
):
    """
    Calculate the chunk size and real-time resolution based on audio properties.
    `preferred_duration` sets the duration of audio sent in each chunk.
    """
    bytes_per_sample = channels * sample_width
    bytes_per_second = sample_rate * bytes_per_sample

    chunk_size = int(bytes_per_second * preferred_duration)
    realtime_resolution = preferred_duration

    return chunk_size, realtime_resolution


async def stream_audio(  # noqa: C901
    output_filename: str,
    filename: str | None,
    url: str,
    encoding: str | None = None,
    channels: int | None = None,
    sample_rate: int | None = None,
    verbose: int = 0,
    message_callback: Callable | None = None,
):
    """
    Stream audio from a file or from the microphone to a Deepgram websocket.
    If `filename` is None, the audio will be streamed from the microphone.
    """
    if filename is None:
        live = True
    else:
        live = False

    # Check to see if the URL is an AIWorks URL, which uses a different output JSON format
    if "aiworks.deepgram.com" in url.lower():
        aiworks = True
    else:
        aiworks = False

    if filename is None:  # `live` is True
        nmessages_to_send = None
        if verbose >= 2:
            output_audio_filename = (
                f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
        else:
            output_audio_filename = None

        microphone = Microphone(output_filename=output_audio_filename)

        async def data_stream() -> AsyncGenerator[bytes, None]:
            print("Streaming audio from microphone...", file=sys.stderr)
            try:
                async for chunk in microphone.stream():
                    yield chunk
            except Exception as e:
                print(f"Error streaming audio: {e}", file=sys.stderr)
                await shutdown(None, microphone)

        realtime_resolution = 0.1

        # The microphone streams 16kHz single-channel linear16 audio.
        # We need to add these params to the URL (if they don't exist already).
        query_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        url_without_query_params = urllib.parse.urlunparse(
            urllib.parse.ParseResult(
                scheme="wss",
                netloc=urllib.parse.urlparse(url).netloc,
                path=urllib.parse.urlparse(url).path,
                params="",
                query="",
                fragment="",
            )
        )
        # Check if this is a v2 API (Flux) request
        is_v2_api = "/v2/" in url or query_params.get("model", [""])[0].startswith(
            "flux"
        )

        # Flux (v2 API) doesn't support the channels parameter
        if "channels" not in query_params and not is_v2_api:
            query_params["channels"] = ["1"]
        if "encoding" not in query_params:
            query_params["encoding"] = ["linear16"]
        if "sample_rate" not in query_params:
            query_params["sample_rate"] = ["16000"]
        url = f"{url_without_query_params}?{urllib.parse.urlencode(query_params, doseq=True)}"
    else:
        # Get information about the audio file.
        # This is used to calculate the chunk size and real-time resolution,
        # which control the pace of the audio sent to the websocket
        # and are used to print useful information.
        channels, sample_width, sample_rate, duration = VerifyAudioFile.analyze(
            filename, encoding, channels, sample_rate, duration=None
        )
        chunk_size, realtime_resolution = calculate_chunk_parameters(
            channels, sample_width, sample_rate
        )
        # Print metadata about the audio file to stderr
        print(
            f"Audio file metadata: {channels} channel(s), {sample_width}-bit sample width, {sample_rate} Hz, {duration} seconds",
            file=sys.stderr,
        )

        microphone = None
        with open(filename, "rb") as f:
            data = f.read()
        nmessages_to_send = len(data) // chunk_size
        if len(data) % chunk_size:
            nmessages_to_send += 1

        async def data_stream() -> AsyncGenerator[bytes, None]:
            for i in range(0, len(data), chunk_size):
                yield data[i : i + chunk_size]

    all_messages = []
    amount_of_audio_sent = 0.0
    try:
        async with websockets.connect(
            url,
            extra_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
        ) as ws:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.ensure_future(shutdown(ws, microphone))
            )
            loop.add_signal_handler(
                signal.SIGTERM, lambda: asyncio.ensure_future(shutdown(ws, microphone))
            )

            ws_open_time = datetime.datetime.now(tz=datetime.timezone.utc)
            custom_open_message = {
                "type": "OpenStream",
                "headers": list(ws.response_headers.raw_items()),
                "received": ws_open_time.isoformat(),
            }
            all_messages.append(custom_open_message)

            request_id = ws.response_headers.get("dg-request-id")
            print(f"Request ID: {request_id}", file=sys.stderr)

            async def sender(ws: websockets.WebSocketClientProtocol):
                nonlocal data
                nonlocal all_messages
                nonlocal ws_open_time
                nonlocal nmessages_to_send
                nonlocal amount_of_audio_sent

                if verbose:
                    # We'll use much more verbose logging than tqdm
                    tqdm_output = open(os.devnull, "w")
                else:
                    tqdm_output = sys.stderr  # type: ignore
                async for c, chunk in asyncstdlib.enumerate(
                    tqdm(
                        iterable=data_stream(),  # type: ignore
                        desc=(
                            f"Streaming {format_time(duration)} of audio"
                            if not live
                            else "Streaming live audio"
                        ),
                        total=nmessages_to_send,
                        leave=False,
                        file=tqdm_output,
                    )
                ):
                    now = datetime.datetime.now(tz=datetime.timezone.utc)
                    time_since_ws_open = (now - ws_open_time).total_seconds()
                    if not live:
                        wait_for = amount_of_audio_sent - time_since_ws_open
                        if wait_for > 0:
                            await asyncio.sleep(wait_for)
                        if verbose >= 2:
                            print(
                                f"Sending {len(chunk)} bytes in message {c}/{nmessages_to_send or 'inf'} "  # type: ignore
                                f"({realtime_resolution} seconds of audio totaling {amount_of_audio_sent + realtime_resolution:0.2f} seconds)"
                                f"at {now.isoformat()} "
                                f"after sleeping for {wait_for} seconds.",
                                file=sys.stderr,
                            )
                    elif verbose >= 2:
                        print(
                            f"Sending {len(chunk)} bytes in message {c} "  # type: ignore
                            f"({realtime_resolution} seconds of audio) "
                            f"at {now.isoformat()}.",
                            file=sys.stderr,
                        )
                    if ws.open:
                        await ws.send(chunk)  # type: ignore
                        amount_of_audio_sent += realtime_resolution

                if ws.open:
                    await ws.send(json.dumps({"type": "CloseStream"}))

            async def receiver(ws):
                nonlocal all_messages
                nonlocal aiworks
                nonlocal amount_of_audio_sent

                async for msg in ws:
                    res = json.loads(msg)

                    # Modify the JSON output to look like Deepram's JSON output if using AIWorks
                    if aiworks:
                        res = res["deepgram_stt"][0]
                        if "channel" in res:
                            res["type"] = "Results"

                    res["received"] = datetime.datetime.now(
                        tz=datetime.timezone.utc
                    ).isoformat()
                    res["audio_cursor"] = amount_of_audio_sent
                    all_messages.append(res)

                    if message_callback:
                        await message_callback(res)

                    if verbose == 1:
                        if res.get("type") == "Results":
                            print(
                                res["channel"]["alternatives"][0]["transcript"],
                                file=sys.stderr,
                            )
                        elif res.get("type") == "TurnInfo":
                            print(
                                f"[{res.get('event', 'Unknown')}] {res.get('transcript', '')}",
                                file=sys.stderr,
                            )
                        else:
                            print(res, file=sys.stderr)
                    elif verbose >= 2:
                        print(f"Received message from Deepgram: {res}", file=sys.stderr)

            await asyncio.gather(
                asyncio.ensure_future(sender(ws)), asyncio.ensure_future(receiver(ws))
            )
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"\n\nHeaders: {e.headers}", file=sys.stderr)
        await shutdown(None, microphone)
        raise e
    except websockets.exceptions.ConnectionClosedError:
        # Allow the finalization logic to complete on abrupt closures,
        # such as when the user presses using Ctrl+C
        print(traceback.format_exc(), file=sys.stderr)
        pass

    with open(output_filename, "w") as of:
        print(json.dumps(all_messages, indent=2, ensure_ascii=False), file=of)
        print(f"Saved transcript to {output_filename}")

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    print(f"Websocket was open for {now - ws_open_time}", file=sys.stderr)
    if filename is not None:
        print(
            f"Total audio duration: {format_time(duration) if duration is not None else 'Unknown'}",
            file=sys.stderr,
        )


async def shutdown(
    websocket: websockets.WebSocketClientProtocol | None, microphone: Microphone | None
):
    if websocket:
        await websocket.close()
    if microphone:
        await microphone.close()


def validate_input(input):
    if os.path.exists(input):
        return input

    raise ValueError(f"{input} is an invalid file path.")


def parse_audio_details_from_url(
    url: str,
) -> tuple[str | None, int | None, int | None]:
    # Example URL:
    # wss://api.deepgram.com/v1/listen?model=nova-2-conversationalai&language=en-US&smart_format=true&punctuate=true&interim_results=true&endpointing=400&utterance_end_ms=1300&encoding=linear16&channels=1&sample_rate=8000

    parsed_url = urllib.parse.urlparse(url)
    _query = urllib.parse.parse_qs(parsed_url.query)
    query: dict[str, str] = {k: v[0] for k, v in _query.items()}  # type: ignore
    encoding: str | None = query.get("encoding")
    if _sample_rate := query.get("sample_rate"):
        sample_rate = int(_sample_rate)
    else:
        sample_rate = None
    if _channels := query.get("channels"):
        channels = int(_channels)
    else:
        # Default to 1 channel if other params are set
        if encoding is not None and sample_rate is not None:
            channels = 1
        else:
            channels = None
    return encoding, sample_rate, channels


@click.group(
    invoke_without_command=True,
    context_settings={"show_default": True},
    help="Stream audio to Deepgram and save all JSON messages to a file.",
)
@click.option(
    "--output",
    "-o",
    required=False,
    help="output file to save JSON results (defaults to input filename with .json extension)",
)
@click.option(
    "--url",
    "-u",
    required=True,
    help="Deepgram websocket URL, e.g. wss://api.deepgram.com/v1/listen?model=nova-2-general&smart_format=true",
)
@click.option(
    "--audio",
    "-f",
    required=False,
    help="audio file to stream (if not streaming from the microphone)",
)
@click.option(
    "--live",
    "-l",
    is_flag=True,
    default=False,
    help="stream audio from the microphone (if not streaming a file)",
)
@click.option(
    "--ui",
    is_flag=True,
    default=False,
    help="launch Textual UI (ignores --output)",
)
@click.option("-v", "--verbose", count=True, help="increase verbosity, e.g. -vvv")
def main(
    output: str,
    url: str,
    audio: str | None,
    live: bool,
    verbose: int = 0,
    ui: bool = False,
):
    if audio is None:
        assert live is True
    else:
        assert live is False
    if live is True:
        assert audio is None
    else:
        assert audio is not None

    # Generate output filename if not provided
    if output is None:
        if audio:
            output = os.path.splitext(os.path.basename(audio))[0] + ".json"
        else:
            output = (
                f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

    if audio and not os.path.exists(audio):
        raise ValueError(f"{audio} does not exist")

    encoding, sample_rate, channels = parse_audio_details_from_url(url)

    if ui:
        from textual_ui import launch_ui

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            launch_ui(audio, url, encoding, channels, sample_rate, verbose)
        )
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            stream_audio(output, audio, url, encoding, channels, sample_rate, verbose)
        )


@main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell):
    """Generate shell completion code"""
    click.echo(
        click.shell_completion.source(  # type: ignore
            prog_name="stream_audio_file.py",
            shell=shell,
        )
    )


if __name__ == "__main__":
    main()
