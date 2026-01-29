from typing import Optional, TypedDict, TypeVar
import json
import datetime
from dataclasses import dataclass

import click
from colorama import Style
from pydantic import BaseModel


T = TypeVar("T")


# ============================================================================
# Configuration Models
# ============================================================================


class DisplayConfig(BaseModel):
    """Configuration for display options"""

    print_channels: bool = False
    print_speakers: bool = False
    print_interim: bool = False
    print_received: bool = False
    print_entities: bool = False
    print_latency: bool = False
    colorize: bool = False
    only_transcript: bool = False


class WordData(BaseModel):
    """Model for word data in transcripts"""

    word: str
    start: float
    end: float
    confidence: float = 1.0
    channel: int = 0
    speaker: Optional[int] = None
    punctuated_word: Optional[str] = None
    entity: Optional[dict] = None

    class Config:
        extra = "allow"  # Allow extra fields from the API


# ============================================================================
# Color Formatting Functions
# ============================================================================


def confidence_color(confidence):
    """
    Convert a number between 0 and 1 to an RGB color where:
    0 maps to red (255, 0, 0) and
    1 maps to white (255, 255, 255)
    """
    confidence = confidence**2
    red = 255
    green = int(255 * confidence)
    blue = int(255 * confidence)
    return red, green, blue


def colorize_latency(
    latency: float,
    min_latency: float,
    max_latency: float,
    use_rich_markup: bool = False,
) -> str:
    """
    Colorize a latency value string based on min/max range.

    Args:
        latency: The latency value in seconds
        min_latency: Minimum latency (maps to white/good)
        max_latency: Maximum latency (maps to red/bad)
        use_rich_markup: If True, use Rich markup format. If False, use ANSI codes.
    """
    text = f"{latency:.3f}s"

    # Convert a latency value to an RGB color where:
    # min_latency maps to white (255, 255, 255) - good
    # max_latency maps to red (255, 0, 0) - bad
    if max_latency <= min_latency:
        # Avoid division by zero; treat as best case
        normalized = 1.0
    else:
        # Normalize: 0 = max (bad), 1 = min (good)
        normalized = 1.0 - (latency - min_latency) / (max_latency - min_latency)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

    # Use same color mapping as confidence
    normalized = normalized**2
    red = 255
    green = int(255 * normalized)
    blue = int(255 * normalized)

    if use_rich_markup:
        return f"[rgb({red},{green},{blue})]{text}[/]"
    else:
        ansi_color = rgb_to_ansi(red, green, blue)
        return f"{ansi_color}{text}{Style.RESET_ALL}"


def rgb_to_ansi(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def colorize_word(text, confidence, use_rich_markup=False):
    """
    Colorize a word based on confidence.

    Args:
        text: The word to colorize
        confidence: Confidence score (0-1)
        use_rich_markup: If True, use Rich markup format. If False, use ANSI codes.
    """
    r, g, b = confidence_color(confidence)

    if use_rich_markup:
        # Rich markup format for Textual
        return f"[rgb({r},{g},{b})]{text}[/]"
    else:
        # ANSI escape codes for terminal
        ansi_color = rgb_to_ansi(r, g, b)
        return f"{ansi_color}{text}{Style.RESET_ALL}"


# ============================================================================
# Time Formatting
# ============================================================================


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


# ============================================================================
# Word Processing Functions
# ============================================================================


class WordType(TypedDict):
    word: str
    start: float
    end: float
    confidence: float
    channel: int
    speaker: Optional[int]
    punctuated_word: Optional[str]
    entity: Optional[dict]


def add_channel_to_word_objects(dg_response: dict):
    if dg_response.get("type") == "TurnInfo":  # Flux
        # For Flux, we don't have explicit channels, so assume channel 0
        for word in dg_response.get("words", []):
            word["channel"] = 0
    elif "results" in dg_response:  # batch
        # Augment the word objects with a "channel" field
        for channel_index, channel in enumerate(dg_response["results"]["channels"]):
            # Always use the first alternative and ignore any others.
            for word in channel["alternatives"][0]["words"]:
                word["channel"] = channel_index
    elif "channel" in dg_response:  # streaming
        channel = dg_response["channel_index"][0]
        for word in dg_response["channel"]["alternatives"][0]["words"]:
            word["channel"] = channel
    else:
        raise ValueError("impossible")
    return dg_response


def remove_empty_sublists(list_of_lists: list[list[T]]) -> list[list[T]]:
    return [sublist for sublist in list_of_lists if len(sublist) > 0]


def split_words_by_speaker(words: list[WordType]) -> list[list[WordType]]:
    """
    Looks at both the speaker and the channel to determine when a speaker change occurs.
    The list of words is split into sublists whenever the speaker or channel changes.
    """
    # Split the word array into subarrays by speaker
    split_by_speaker: list[list[WordType]] = []
    speaker = None
    channel = None
    for word in words:
        word_speaker = word.get("speaker", None)
        word_channel = word["channel"]
        if word_speaker != speaker or word_channel != channel:
            split_by_speaker.append([])
        split_by_speaker[-1].append(word)
        speaker = word_speaker
        channel = word_channel
    # Remove empty word arrays from speaker array
    split_by_speaker = remove_empty_sublists(split_by_speaker)
    return split_by_speaker


def split_words_into_sentences(words: list[WordType]) -> list[list[WordType]]:
    split: list[list[WordType]] = [[]]
    # Split at a period, question mark, or exclamation point.
    for word in words:
        split[-1].append(word)
        _word = word.get("punctuated_word") or word["word"]
        if any(p in _word for p in [".", "?", "!"]):
            split.append([])
    del words
    split = remove_empty_sublists(split)
    return split


def add_entities_to_word_objects(
    words: list[WordType], entities: list[dict]
) -> list[WordType]:
    """
    Add the entity information to the words.
    """
    for entity in entities:
        for i in range(entity["start_word"], entity["end_word"]):
            words[i]["entity"] = entity
    return words


def apply_entity_brackets(word_array: list[WordType]) -> None:
    """
    Modifies word_array in place to add [ENTITY: text] formatting.
    Handles nested entities and fixes punctuation placement.
    """
    # Track active entities to handle nesting
    active_entities = {}

    for i, word in enumerate(word_array):
        if word is None:
            continue

        # Get current word text
        word_text: str = word.get("punctuated_word") or word["word"]

        # Check if this word has an entity
        current_entity = word.get("entity", None)
        current_entity_key = current_entity["label"] if current_entity else None

        # Check next word to see if entity continues
        next_entity = None
        if i + 1 < len(word_array) and word_array[i + 1] is not None:
            next_entity = word_array[i + 1].get("entity", None)

        # Check if this is the start of a new entity
        if current_entity and current_entity_key not in active_entities:
            # Add opening bracket
            word_text = f"[{current_entity_key}: {word_text}"
            active_entities[current_entity_key] = True

        # Check if this is the end of the current entity
        if current_entity and (
            next_entity is None or next_entity["label"] != current_entity["label"]
        ):
            # Add closing bracket
            word_text = f"{word_text}]"
            if current_entity_key in active_entities:
                del active_entities[current_entity_key]

        # Update the word
        word["punctuated_word"] = word_text

    # Close any remaining open entities at the end
    if len(word_array) > 0 and word_array[-1] is not None:
        last_word = word_array[-1]
        last_word_text = last_word.get("punctuated_word") or last_word["word"]
        for _ in active_entities:
            last_word_text += "]"
        last_word["punctuated_word"] = last_word_text

    # Fix any .] or ,] punctuation issues.
    punctuation = [".", ",", "!", "?", ":", ";"]
    for word in word_array:
        word_text = word.get("punctuated_word") or word["word"]
        for p in punctuation:
            if word_text.endswith(f"{p}]"):
                word_text = word_text[:-2] + "]" + p
                word["punctuated_word"] = word_text


# ============================================================================
# Speaker Management
# ============================================================================


class SpeakerAggregator:
    def __init__(self) -> None:
        self.speaker_labels: dict[tuple[int, int | None], int] = {}

    def same_speaker_in_word_array(
        self,
        word_array: list[WordType],
        different_channels_are_different_speakers: bool,
    ) -> bool:
        """Returns "True" if all word-level speaker tags are the same."""
        labels = set()
        for word in word_array:
            labels.add(
                self.get_speaker_label(word, different_channels_are_different_speakers)
            )
        return len(labels) == 1

    def get_speaker_labels(
        self,
        word_array: list[WordType],
        different_channels_are_different_speakers: bool,
    ) -> str:
        """
        Get the speaker labels for the given word array.
        Multiple speaker labels are returned if the speaker changes within the word array.
        """
        if len(word_array) == 0:
            return "-"
        labels = sorted(
            set(
                self.get_speaker_label(word, different_channels_are_different_speakers)
                for word in word_array
            )
        )
        return "+".join(str(label) for label in labels)

    def get_speaker_label(
        self, word: WordType, different_channels_are_different_speakers: bool
    ) -> str:
        """
        Get the speaker label for the given word. Unique speaker labels are generated on-demand.
        The mapping is created on-the-fly as the words are processed using the channel and speaker.
        """
        if different_channels_are_different_speakers:
            # Use the channel and speaker to create a unique speaker label
            speaker_id = (word["channel"], word.get("speaker", None))
        else:
            speaker_id = (0, word.get("speaker", None))
        if speaker_id not in self.speaker_labels:
            self.speaker_labels[speaker_id] = len(self.speaker_labels)
        return str(self.speaker_labels[speaker_id])


# ============================================================================
# Transcript Line Building
# ============================================================================


@dataclass
class TranscriptLine:
    """Helper class to build formatted transcript lines"""

    config: DisplayConfig
    words_str: str
    start: float
    end: float
    events: tuple[str, ...]
    speaker_label: str = ""
    channel: Optional[int | tuple[int, ...]] = None
    received: Optional[datetime.datetime] = None
    latency: Optional[float] = None
    eot_latency: Optional[float] = None

    def format(self) -> str:
        """Build the final formatted line"""
        parts = []

        # Add received timestamp if configured
        if self.config.print_received and self.received is not None:
            received_str = f"[{self.received.strftime('%H:%M:%S.%f')}]"
            parts.append(received_str)

        # Add interim latency if configured and available
        if self.config.print_latency and self.latency is not None:
            parts.append(f"[latency={self.latency:.3f}s]")

        # Add EOT latency if configured and available
        if self.config.print_latency and self.eot_latency is not None:
            parts.append(f"[eot_latency={self.eot_latency:.3f}s]")

        # Add duration
        duration = f"[{format_time(self.start)} - {format_time(self.end)}]"
        parts.append(duration)

        # Add speaker if configured
        if self.config.print_speakers and self.speaker_label:
            parts.append(f"[Speaker {self.speaker_label}]")

        # Add channel if configured
        if self.config.print_channels and self.channel is not None:
            parts.append(f"[Channel {self.channel}]")

        # Add events
        for event in self.events:
            parts.append(f"[{event}]")

        # Combine all parts
        line = " ".join(parts) + f": {self.words_str}"

        # Clean up spacing
        while "  " in line:
            line = line.replace("  ", " ")
        while " :" in line:
            line = line.replace(" :", ":")

        return line.strip()


def str_from_word_array(
    streaming: bool,
    received: datetime.datetime | None,
    latency: float | None,
    eot_latency: float | None,
    word_array: list[WordType],
    channel: int | tuple[int, ...],
    start: float,
    end: float,
    events: tuple[str, ...],
    allow_multiple_speakers: bool,
    speaker_aggregator: SpeakerAggregator,
    config: DisplayConfig,
    use_rich_markup: bool = False,
) -> str | None:
    """
    Convert the word array and associated metadata into a formatted string.
    """
    # Return without printing if we are not printing interim results and is_final=false .
    if streaming and not config.print_interim:
        if "InterimResult" in events or "Update" in events or "StartOfTurn" in events:
            return None

    # Apply entity brackets if configured
    if config.print_entities:
        apply_entity_brackets(word_array)

    # Build the words string with optional colorization
    if config.colorize:
        words_str = " ".join(
            [
                colorize_word(
                    word.get("punctuated_word", word["word"]),
                    word["confidence"],
                    use_rich_markup=use_rich_markup,
                )
                for word in word_array
            ]
        )
    else:
        words_str = " ".join(
            [word.get("punctuated_word") or word["word"] for word in word_array]
        )

    # Determine speaker configuration
    # If print_speaker is true and print_channels is true, then don't treat channels are different speakers.
    if config.print_speakers and config.print_channels:
        different_channels_are_different_speakers = False
    # If print_speakers is true and print_channels is false, then treat channels as different speakers.
    elif config.print_speakers and not config.print_channels:
        different_channels_are_different_speakers = True
    # If print_speakers is false and print_channels is true, then don't treat channels as different speakers.
    elif not config.print_speakers and config.print_channels:
        different_channels_are_different_speakers = False
    # If print_speakers is false and print_channels is false, then don't treat channels as different speakers.
    elif not config.print_speakers and not config.print_channels:
        different_channels_are_different_speakers = False
    else:
        raise ValueError("impossible")

    # Get speaker label(s)
    if not allow_multiple_speakers:
        if len(word_array) > 0:
            # Ensure all words are from the same speaker.
            assert speaker_aggregator.same_speaker_in_word_array(
                word_array, different_channels_are_different_speakers
            )
            # Get the single speaker label.
            speaker_label = speaker_aggregator.get_speaker_label(
                word_array[0], different_channels_are_different_speakers
            )
        else:  # no words!
            speaker_label = "0"  # Default for empty arrays
    else:
        speaker_label = speaker_aggregator.get_speaker_labels(
            word_array, different_channels_are_different_speakers
        )

    # Build and format the transcript line
    line_builder = TranscriptLine(
        words_str=words_str,
        start=start,
        end=end,
        speaker_label=speaker_label if config.print_speakers else "",
        channel=channel,
        events=events,
        received=received,
        latency=latency,
        eot_latency=eot_latency,
        config=config,
    )

    return line_builder.format()


# ============================================================================
# EOT Latency Tracking (End-of-Turn Latency)
# ============================================================================


class EOTLatencyAggregator:
    """
    Calculates End-of-Turn (EOT) latency for streaming transcription.

    EOT latency measures the time between receiving the last interim result
    and receiving a finalizing event. This represents the additional delay,
    on top of interim result latency, required to identify that the user has
    finished speaking. This is critical for voice agents because they cannot
    respond until they know the user is done.

    Supported EOT events:
    - Nova: is_final=true, speech_final=true, UtteranceEnd
    - Flux: EndOfTurn, EagerEndOfTurn
    """

    def __init__(self) -> None:
        # Track the last interim result time per channel
        self.last_interim_time: dict[int, datetime.datetime] = {}
        # Store measurements as (event_type, latency_seconds)
        self.measurements: list[tuple[str, float]] = []

    def record_interim(self, message: dict, received: datetime.datetime) -> None:
        """Record the time of an interim result (Nova or Flux)."""
        # Nova: Results with is_final=false
        if message.get("type") == "Results":
            if message.get("is_final", False):
                return  # Not an interim result
            channel = message.get("channel_index", [0])[0]
            self.last_interim_time[channel] = received

        # Flux: TurnInfo with Update or StartOfTurn event
        elif message.get("type") == "TurnInfo":
            event = message.get("event")
            if event in ("Update", "StartOfTurn"):
                # Flux doesn't have channels, use 0
                self.last_interim_time[0] = received

    def calculate_eot_latency(
        self, message: dict, received: datetime.datetime
    ) -> tuple[str, float] | None:
        """
        Calculate EOT latency for end-of-turn events.

        Returns (event_type, latency_seconds) or None if not an EOT event
        or if there was no preceding interim result.
        """
        event_type: str | None = None
        channel: int = 0

        # Nova: Check for EOT events in Results messages
        if message.get("type") == "Results":
            channel = message.get("channel_index", [0])[0]
            if message.get("speech_final", False):
                event_type = "speech_final"
            elif message.get("is_final", False):
                event_type = "is_final"

        # Nova: UtteranceEnd message
        elif message.get("type") == "UtteranceEnd":
            channel = message.get("channel", [0])[0]
            event_type = "UtteranceEnd"

        # Flux: TurnInfo with EndOfTurn or EagerEndOfTurn event
        elif message.get("type") == "TurnInfo":
            event = message.get("event")
            if event == "EndOfTurn":
                event_type = "EndOfTurn"
                channel = 0  # Flux doesn't have channels
            elif event == "EagerEndOfTurn":
                event_type = "EagerEndOfTurn"
                channel = 0

        if event_type is None:
            return None

        # Get the last interim time for this channel
        last_interim = self.last_interim_time.get(channel)
        if last_interim is None:
            # No preceding interim result recorded
            return None

        # Calculate latency
        eot_latency = (received - last_interim).total_seconds()

        # Only record positive latencies (negative would indicate timing issues)
        if eot_latency >= 0:
            self.measurements.append((event_type, eot_latency))

        # Clear the last interim time for this channel after a definitive EOT event
        # - Nova: speech_final and UtteranceEnd are definitive; is_final may have more
        # - Flux: EndOfTurn is definitive; EagerEndOfTurn may be followed by TurnResumed
        if event_type in ("speech_final", "UtteranceEnd", "EndOfTurn"):
            self.last_interim_time.pop(channel, None)

        return (event_type, eot_latency)

    def get_stats(self) -> dict:
        """Return EOT latency statistics."""
        if not self.measurements:
            return {"count": 0}

        latencies = [m[1] for m in self.measurements]
        sorted_latencies = sorted(latencies)

        def percentile(p: float) -> float:
            idx = int(len(sorted_latencies) * p)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

        return {
            "min": min(latencies),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(latencies),
            "count": len(self.measurements),
            "by_event": self._stats_by_event(),
        }

    def _stats_by_event(self) -> dict[str, dict]:
        """Get stats broken down by event type."""
        from collections import defaultdict

        by_event: dict[str, list[float]] = defaultdict(list)
        for event_type, latency in self.measurements:
            by_event[event_type].append(latency)

        def percentile(sorted_vals: list[float], p: float) -> float:
            idx = int(len(sorted_vals) * p)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        result = {}
        for event_type, latencies in by_event.items():
            sorted_latencies = sorted(latencies)
            result[event_type] = {
                "min": min(latencies),
                "p50": percentile(sorted_latencies, 0.50),
                "p95": percentile(sorted_latencies, 0.95),
                "p99": percentile(sorted_latencies, 0.99),
                "max": max(latencies),
                "count": len(latencies),
            }
        return result

    def format_summary(self) -> str:
        """Return formatted summary string."""
        stats = self.get_stats()
        if stats["count"] == 0:
            return "No EOT latency measurements (requires interim_results=true and EOT events)"

        lines = [
            f"EOT Latency: min={stats['min']:.3f}s, p50={stats['p50']:.3f}s, "
            f"p95={stats['p95']:.3f}s, p99={stats['p99']:.3f}s, "
            f"max={stats['max']:.3f}s ({stats['count']} events)"
        ]

        for event_type, event_stats in stats["by_event"].items():
            lines.append(
                f"  {event_type}: min={event_stats['min']:.3f}s, p50={event_stats['p50']:.3f}s, "
                f"p95={event_stats['p95']:.3f}s, p99={event_stats['p99']:.3f}s, "
                f"max={event_stats['max']:.3f}s ({event_stats['count']} events)"
            )

        return "\n".join(lines)

    def colorize_lines(
        self, lines: list[str], use_rich_markup: bool = False
    ) -> list[str]:
        """
        Post-process transcript lines to colorize EOT latency values.

        Replaces [eot_latency=X.XXXs] with colorized version based on min/max.
        """
        import re

        stats = self.get_stats()
        if stats["count"] == 0:
            return lines

        min_lat = stats["min"]
        max_lat = stats["max"]

        # If the worst latency is less than 400ms, that's still really good.
        # We will manually increase the worst case latency to 400ms in these cases.
        if max_lat <= 0.4:
            max_lat = 0.4

        # Pattern to match [eot_latency=0.123s]
        pattern = re.compile(r"\[eot_latency=(\d+\.\d+)s\]")

        colorized_lines = []
        for line in lines:

            def replace_latency(match):
                latency_val = float(match.group(1))
                colored = colorize_latency(
                    latency_val, min_lat, max_lat, use_rich_markup
                )
                return f"[eot_latency={colored}]"

            colorized_lines.append(pattern.sub(replace_latency, line))

        return colorized_lines


# ============================================================================
# Interim Latency Tracking (Message Latency per Deepgram methodology)
# ============================================================================


class LatencyAggregator:
    """
    Calculates message latency following Deepgram's methodology.

    Latency = audio_cursor - transcript_cursor

    Where:
        audio_cursor = seconds of audio sent to Deepgram (from message)
        transcript_cursor = start + duration from the response

    Per Deepgram docs, latency is measured using INTERIM results because
    they arrive faster and reflect actual real-time latency.

    Supported interim message types:
    - Nova: Results with is_final=false
    - Flux: TurnInfo with event=Update or event=TurnResumed

    Reference: https://developers.deepgram.com/docs/measuring-streaming-latency
    """

    def __init__(self) -> None:
        self.measurements: list[float] = []

    def calculate_latency(self, message: dict) -> float | None:
        """
        Calculate latency for an interim result message (Nova or Flux).

        Returns latency in seconds, or None if not applicable.
        """
        audio_cursor: float | None = None
        transcript_cursor: float | None = None

        # Nova: Results messages
        if message.get("type") == "Results":
            # Only measure interim results
            if message.get("is_final", True):
                return None

            audio_cursor = message.get("audio_cursor")
            start = message.get("start", 0)
            duration = message.get("duration", 0)

            if duration <= 0:
                return None

            transcript_cursor = start + duration

        # Flux: TurnInfo messages with Update or TurnResumed events
        elif message.get("type") == "TurnInfo":
            event = message.get("event")
            if event not in ("Update", "TurnResumed"):
                return None

            audio_cursor = message.get("audio_cursor")
            audio_window_end = message.get("audio_window_end", 0)

            if audio_window_end <= 0:
                return None

            transcript_cursor = audio_window_end

        else:
            return None

        # Need audio_cursor to calculate latency
        if audio_cursor is None:
            return None

        if transcript_cursor is None:
            return None

        cur_latency = audio_cursor - transcript_cursor

        # Record measurement
        self.measurements.append(cur_latency)

        return cur_latency

    def get_p50(self) -> float | None:
        """Calculate p50 latency."""
        if not self.measurements:
            return None
        sorted_measurements = sorted(self.measurements)
        idx = int(len(sorted_measurements) * 0.50)
        idx = min(idx, len(sorted_measurements) - 1)
        return sorted_measurements[idx]

    def get_p95(self) -> float | None:
        """Calculate p95 latency."""
        if not self.measurements:
            return None
        sorted_measurements = sorted(self.measurements)
        idx = int(len(sorted_measurements) * 0.95)
        idx = min(idx, len(sorted_measurements) - 1)
        return sorted_measurements[idx]

    def get_p99(self) -> float | None:
        """Calculate p99 latency."""
        if not self.measurements:
            return None
        sorted_measurements = sorted(self.measurements)
        idx = int(len(sorted_measurements) * 0.99)
        idx = min(idx, len(sorted_measurements) - 1)
        return sorted_measurements[idx]

    def get_stats(self) -> dict:
        """Return latency statistics."""
        if not self.measurements:
            return {
                "min": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "count": 0,
            }

        p50 = self.get_p50()
        p95 = self.get_p95()
        p99 = self.get_p99()

        return {
            "min": min(self.measurements) if self.measurements else None,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "max": max(self.measurements) if self.measurements else None,
            "count": len(self.measurements),
        }

    def format_summary(self) -> str:
        """Return formatted summary string."""
        stats = self.get_stats()
        if stats["count"] == 0:
            return "No latency measurements (requires interim_results=true and audio_cursor in messages)"
        return (
            f"Message Latency: min={round(stats['min'], 3):.3f}s, p50={round(stats['p50'], 3):.3f}s, p95={round(stats['p95'], 3):.3f}s, p99={round(stats['p99'], 3):.3f}s, "
            f"max={round(stats['max'], 3):.3f}s ({stats['count']} measurements)"
        )

    def colorize_lines(
        self, lines: list[str], use_rich_markup: bool = False
    ) -> list[str]:
        """
        Post-process transcript lines to colorize latency values.

        Replaces [latency=X.XXXs] with colorized version based on min/max.
        """
        import re

        stats = self.get_stats()
        min_latency = stats["min"]
        max_latency = stats["max"]

        if not self.measurements or min_latency is None or max_latency is None:
            return lines

        min_lat = min_latency
        max_lat = max_latency

        # If the worst latency is less than 400ms, that's still really good.
        # We will manually increase the worst case latency to 400ms in these cases.
        if max_lat <= 0.4:
            max_lat = 0.4

        # Pattern to match [latency=0.123s]
        pattern = re.compile(r"\[latency=(\d+\.\d+)s\]")

        colorized_lines = []
        for line in lines:

            def replace_latency(match):
                latency_val = float(match.group(1))
                colored = colorize_latency(
                    latency_val, min_lat, max_lat, use_rich_markup
                )
                return f"[latency={colored}]"

            colorized_lines.append(pattern.sub(replace_latency, line))

        return colorized_lines


# ============================================================================
# Response Metrics (TTFT and Update Frequency)
# ============================================================================


class ResponseMetricsAggregator:
    """
    Tracks Time-to-First-Transcript (TTFT) and Update Frequency metrics.

    TTFT: Wall-clock time from first audio sent (OpenStream) to first
    transcript message received (Results or TurnInfo). This includes empty
    transcripts, as they still indicate Deepgram has processed audio.

    Update Frequency: Number of interim/update messages per second of audio.
    This captures how "responsive" the transcription feels to users, as more
    frequent updates create a more fluid real-time experience.
    """

    def __init__(self) -> None:
        # TTFT tracking
        self.first_audio_sent_time: datetime.datetime | None = None
        self.first_transcript_received_time: datetime.datetime | None = None

        # Update frequency tracking
        self.interim_message_count: int = 0
        self.total_audio_duration: float = 0.0

    def record_stream_start(self, received: datetime.datetime) -> None:
        """Record when audio streaming begins (OpenStream message)."""
        if self.first_audio_sent_time is None:
            self.first_audio_sent_time = received

    def record_transcript_message(
        self, message: dict, received: datetime.datetime
    ) -> None:
        """
        Record a transcript message for TTFT and update frequency tracking.

        Args:
            message: The Results or TurnInfo message
            received: When the message was received
        """
        msg_type = message.get("type")

        # Track first transcript for TTFT (including empty transcripts)
        if self.first_transcript_received_time is None:
            if msg_type in ("Results", "TurnInfo"):
                self.first_transcript_received_time = received

        # Track interim messages for update frequency
        if msg_type == "Results":
            # Nova: Count interim results (is_final=false)
            if not message.get("is_final", True):
                self.interim_message_count += 1
            # Track audio duration from the last message
            audio_cursor = message.get("audio_cursor")
            if audio_cursor is not None and audio_cursor > self.total_audio_duration:
                self.total_audio_duration = audio_cursor

        elif msg_type == "TurnInfo":
            # Flux: Count Update and TurnResumed events as interim messages
            event = message.get("event")
            if event in ("Update", "TurnResumed"):
                self.interim_message_count += 1
            # Track audio duration
            audio_cursor = message.get("audio_cursor")
            if audio_cursor is not None and audio_cursor > self.total_audio_duration:
                self.total_audio_duration = audio_cursor

    def get_ttft(self) -> float | None:
        """
        Calculate Time-to-First-Transcript in seconds.

        Returns None if either timestamp is missing.
        """
        if (
            self.first_audio_sent_time is None
            or self.first_transcript_received_time is None
        ):
            return None
        return (
            self.first_transcript_received_time - self.first_audio_sent_time
        ).total_seconds()

    def get_update_frequency(self) -> float | None:
        """
        Calculate update frequency (interim messages per second of audio).

        Returns None if no audio duration is recorded.
        """
        if self.total_audio_duration <= 0:
            return None
        return self.interim_message_count / self.total_audio_duration

    def get_stats(self) -> dict:
        """Return all response metrics."""
        return {
            "ttft": self.get_ttft(),
            "update_frequency": self.get_update_frequency(),
            "interim_count": self.interim_message_count,
            "audio_duration": self.total_audio_duration,
        }


# ============================================================================
# Main Processing Functions
# ============================================================================


def formatted_batch_transcript(
    dg_response: dict,
    config: DisplayConfig,
) -> str:
    dg_response = add_channel_to_word_objects(dg_response)

    # Flatten/combine the word arrays across all channels
    words: list[WordType] = [
        w
        for c in dg_response["results"]["channels"]
        # Always use the first alternative and ignore any others.
        for w in c["alternatives"][0]["words"]
    ]
    # Re-sort the word arrays by the words' start time
    words.sort(key=lambda word: (word["start"] + word["end"]) / 2)

    # Add entities to the words, if desired
    if config.print_entities:
        entities = []
        for channel in dg_response["results"]["channels"]:
            entities.extend(channel["alternatives"][0].get("entities", []))
        words = add_entities_to_word_objects(words, entities)

    # Split the word array into subarrays by speaker
    word_arrays = split_words_by_speaker(words)

    # If there is only one speaker, break the word array into sentences so it's easier to read.
    if len(word_arrays) == 1:
        word_arrays = split_words_into_sentences(word_arrays[0])

    # Create the speaker-by-speaker transcripts
    speaker_aggregator = SpeakerAggregator()
    single_line_transcripts: list[str] = []
    for word_array in word_arrays:
        start = word_array[0]["start"]
        end = word_array[-1]["end"]
        channels = tuple(list(set(word["channel"] for word in word_array)))
        if len(channels) == 1:
            channel = channels[0]
        else:
            channel = channels
        single_line_transcript = str_from_word_array(
            streaming=False,
            received=None,
            latency=None,
            eot_latency=None,
            word_array=word_array,
            channel=channel,
            start=start,
            end=end,
            events=(),
            allow_multiple_speakers=False,
            speaker_aggregator=speaker_aggregator,
            config=config,
        )
        if single_line_transcript is not None:
            single_line_transcripts.append(single_line_transcript)
    transcript = "\n".join(single_line_transcripts)
    return transcript


class StreamingTranscriptPrinter:
    @classmethod
    def render_open_stream(
        cls,
        message: dict,
        received: datetime.datetime | None,
        config: DisplayConfig,
    ) -> str:
        if received is not None:
            received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
            request_id = {k: v for k, v in message.get("headers", [])}.get(
                "dg-request-id"
            )
            return f"{received_str} [OpenStream <request_id={request_id}>]"
        else:
            return "[OpenStream]"

    @classmethod
    def render_connected(
        cls,
        message: dict,
        received: datetime.datetime | None,
        config: DisplayConfig,
    ) -> str:
        if received is not None:
            received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
            request_id = message.get("request_id")
            return f"{received_str} [Connected <request_id={request_id}>]"
        else:
            return "[Connected]"

    @classmethod
    def render_metadata(
        cls,
        message: dict,
        received: datetime.datetime | None,
        config: DisplayConfig,
    ) -> str:
        if received is not None:
            received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
            return f"{received_str} [Metadata/CloseStream]"
        else:
            return "[Metadata/CloseStream]"

    @classmethod
    def render_results(
        cls,
        message: dict,
        received: datetime.datetime | None,
        latency: float | None,
        eot_latency: float | None,
        config: DisplayConfig,
        speaker_aggregator: SpeakerAggregator,
        use_rich_markup: bool = False,
    ) -> str | None:
        message = add_channel_to_word_objects(message)
        word_array: list[WordType] = message["channel"]["alternatives"][0]["words"]
        channel = message["channel_index"][0]

        # Create the speaker-by-speaker transcripts
        start = message["start"]
        end = message["start"] + message["duration"]

        events = []
        if message.get("is_final", False):
            events.append("IsFinal")
        if message.get("speech_final", False):
            events.append("SpeechFinal")
        if not message.get("is_final", False) and not message.get(
            "speech_final", False
        ):
            events.append("InterimResult")
        line = str_from_word_array(
            streaming=True,
            received=received,
            latency=latency,
            eot_latency=eot_latency,
            word_array=word_array,
            channel=channel,
            start=start,
            end=end,
            events=tuple(events),
            allow_multiple_speakers=True,
            speaker_aggregator=speaker_aggregator,
            config=config,
            use_rich_markup=use_rich_markup,
        )
        return line

    @classmethod
    def render_utterance_end(
        cls,
        message: dict,
        received: datetime.datetime | None,
        eot_latency: float | None,
        config: DisplayConfig,
    ) -> str:
        channel = message["channel"][0]
        last_word_end = (
            format_time(message["last_word_end"])
            if message["last_word_end"] > 0
            else str(message["last_word_end"])
        )
        if config.print_channels:
            channel_str = f"[Channel {channel}]"
        else:
            channel_str = ""
        if config.print_received and received is not None:
            received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
        else:
            received_str = ""
        if config.print_latency and eot_latency is not None:
            eot_str = f"[eot_latency={eot_latency:.3f}s]"
        else:
            eot_str = ""
        line = f"{received_str} {eot_str} {channel_str} [UtteranceEnd]: last_word_end={last_word_end}"
        while "  " in line:
            line = line.replace("  ", " ")
        while " :" in line:
            line = line.replace(" :", ":")
        line = line.strip()
        return line

    @classmethod
    def render_start_of_turn(
        cls,
        message: dict,
        received: datetime.datetime | None,
        config: DisplayConfig,
    ) -> str:
        channel = message["channel"][0]
        if config.print_channels:
            channel_str = f"[Channel {channel}]"
        else:
            channel_str = ""
        if config.print_received and received is not None:
            received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
        else:
            received_str = ""
        timestamp = format_time(message["timestamp"])
        line = f"{received_str} [{timestamp}] {channel_str} [StartOfTurn]"
        while "  " in line:
            line = line.replace("  ", " ")
        while " :" in line:
            line = line.replace(" :", ":")
        line = line.strip()
        return line

    @classmethod
    def render_turn_info(
        cls,
        message: dict,
        received: datetime.datetime | None,
        latency: float | None,
        eot_latency: float | None,
        config: DisplayConfig,
        speaker_aggregator: SpeakerAggregator,
        use_rich_markup: bool = False,
    ) -> str | None:
        # Handle Flux TurnInfo messages
        message = add_channel_to_word_objects(message)
        word_array: list[WordType] = message.get("words", [])
        channel = 0  # Flux does not have channels in TurnInfo messages

        start = message.get("audio_window_start", 0)
        end = message.get("audio_window_end", 0)
        event = message.get("event", None)

        events = (event,) if event else ()
        line = str_from_word_array(
            received=received,
            latency=latency,
            eot_latency=eot_latency,
            streaming=True,
            word_array=word_array,
            channel=channel,
            start=start,
            end=end,
            events=events,
            allow_multiple_speakers=True,
            speaker_aggregator=speaker_aggregator,
            config=config,
            use_rich_markup=use_rich_markup,
        )
        return line

    @classmethod
    def get_transcript(
        cls,
        messages: list[dict],
        config: DisplayConfig,
    ) -> str:
        speaker_aggregator = SpeakerAggregator()

        # Initialize latency aggregators if latency printing is enabled
        latency_aggregator = LatencyAggregator() if config.print_latency else None
        eot_latency_aggregator = (
            EOTLatencyAggregator() if config.print_latency else None
        )
        response_metrics_aggregator = (
            ResponseMetricsAggregator() if config.print_latency else None
        )

        single_line_transcripts: list[str] = []
        for message in messages:
            # Parse received time
            received_iso = message.get("received", None)
            if received_iso is None:
                received = None
            else:
                received = datetime.datetime.fromisoformat(received_iso)

            # Calculate interim latency for Results and TurnInfo messages
            latency = None
            if latency_aggregator is not None:
                if message.get("type") in ("Results", "TurnInfo"):
                    latency = latency_aggregator.calculate_latency(message)

            # Track interim results and calculate EOT latency
            eot_latency = None
            if eot_latency_aggregator is not None and received is not None:
                # Record interim results (Nova and Flux)
                eot_latency_aggregator.record_interim(message, received)
                # Calculate EOT latency for finalizing events
                eot_result = eot_latency_aggregator.calculate_eot_latency(
                    message, received
                )
                if eot_result is not None:
                    eot_latency = eot_result[1]

            # Handle UtteranceEnd messages separately because they have a different type.
            line: str | None
            if message.get("type") == "UtteranceEnd":
                line = cls.render_utterance_end(
                    message=message,
                    received=received,
                    eot_latency=eot_latency,
                    config=config,
                )
                single_line_transcripts.append(line)

            # Print the OpenStream message if it exists.
            elif message.get("type") == "OpenStream":
                # Record stream start time for TTFT calculation
                if response_metrics_aggregator is not None and received is not None:
                    response_metrics_aggregator.record_stream_start(received)
                line = cls.render_open_stream(
                    message=message,
                    received=received,
                    config=config,
                )
                single_line_transcripts.append(line)
            # Print the Connected message (Flux)
            elif message.get("type") == "Connected":
                line = cls.render_connected(
                    message=message,
                    received=received,
                    config=config,
                )
                single_line_transcripts.append(line)

            # Print the Metadata/CloseStream message if it exists.
            elif message.get("type") == "Metadata":
                line = cls.render_metadata(
                    message=message,
                    received=received,
                    config=config,
                )
                single_line_transcripts.append(line)

            elif message.get("type") == "Results":
                # Record for TTFT and update frequency tracking
                if response_metrics_aggregator is not None and received is not None:
                    response_metrics_aggregator.record_transcript_message(
                        message, received
                    )
                line = cls.render_results(
                    message=message,
                    received=received,
                    latency=latency,
                    eot_latency=eot_latency,
                    config=config,
                    speaker_aggregator=speaker_aggregator,
                )
                if line is not None:
                    single_line_transcripts.append(line)

            elif message.get("type") == "StartOfTurn":
                line = cls.render_start_of_turn(
                    message=message,
                    received=received,
                    config=config,
                )
                single_line_transcripts.append(line)

            elif message.get("type") == "TurnInfo":
                # Record for TTFT and update frequency tracking
                if response_metrics_aggregator is not None and received is not None:
                    response_metrics_aggregator.record_transcript_message(
                        message, received
                    )
                line = cls.render_turn_info(
                    message=message,
                    received=received,
                    latency=latency,
                    eot_latency=eot_latency,
                    config=config,
                    speaker_aggregator=speaker_aggregator,
                )
                if line is not None:
                    single_line_transcripts.append(line)
            else:
                print("Skipping message of type:", message.get("type"))

        # Colorize latency values if enabled
        if latency_aggregator is not None and config.colorize:
            single_line_transcripts = latency_aggregator.colorize_lines(
                single_line_transcripts, use_rich_markup=False
            )

        # Colorize EOT latency values if enabled
        if eot_latency_aggregator is not None and config.colorize:
            single_line_transcripts = eot_latency_aggregator.colorize_lines(
                single_line_transcripts, use_rich_markup=False
            )

        # Append response metrics summary (TTFT and Update Frequency) if enabled
        if response_metrics_aggregator is not None:
            stats = response_metrics_aggregator.get_stats()
            single_line_transcripts.append("")
            if stats["ttft"] is not None:
                single_line_transcripts.append(
                    f"Time-to-First-Transcript: {stats['ttft']:.3f}s"
                )
            if stats["update_frequency"] is not None:
                single_line_transcripts.append(
                    f"Update Frequency: {stats['update_frequency']:.2f} updates/sec "
                    f"({stats['interim_count']} updates over {stats['audio_duration']:.1f}s of audio)"
                )

        # Append latency summaries if enabled
        if latency_aggregator is not None:
            single_line_transcripts.append("")
            single_line_transcripts.append(latency_aggregator.format_summary())

        if eot_latency_aggregator is not None:
            single_line_transcripts.append("")
            single_line_transcripts.append(eot_latency_aggregator.format_summary())

        transcript = "\n".join(single_line_transcripts)
        return transcript


class StreamingFormatter:
    """Stateful formatter for processing messages one at a time"""

    def __init__(
        self,
        config: DisplayConfig,
        use_rich_markup: bool = False,
    ):
        self.config = config
        self.speaker_aggregator = SpeakerAggregator()
        self.use_rich_markup = use_rich_markup

        # Initialize latency aggregators if latency printing is enabled
        self.latency_aggregator = LatencyAggregator() if config.print_latency else None
        self.eot_latency_aggregator = (
            EOTLatencyAggregator() if config.print_latency else None
        )
        self.response_metrics_aggregator = (
            ResponseMetricsAggregator() if config.print_latency else None
        )

    def format_message(self, message: dict) -> str | None:
        """Format a single message and return the formatted string"""

        # Parse received time
        received = None
        received_iso = message.get("received", None)
        if received_iso is not None:
            received = datetime.datetime.fromisoformat(received_iso)

        # Calculate interim latency for Results and TurnInfo messages
        latency = None
        if self.latency_aggregator is not None:
            if message.get("type") in ("Results", "TurnInfo"):
                latency = self.latency_aggregator.calculate_latency(message)

        # Track interim results and calculate EOT latency
        eot_latency = None
        if self.eot_latency_aggregator is not None and received is not None:
            # Record interim results (Nova and Flux)
            self.eot_latency_aggregator.record_interim(message, received)
            # Calculate EOT latency for finalizing events
            eot_result = self.eot_latency_aggregator.calculate_eot_latency(
                message, received
            )
            if eot_result is not None:
                eot_latency = eot_result[1]

        # Handle different message types
        msg_type = message.get("type")

        if msg_type == "OpenStream":
            # Record stream start time for TTFT calculation
            if self.response_metrics_aggregator is not None and received is not None:
                self.response_metrics_aggregator.record_stream_start(received)
            return StreamingTranscriptPrinter.render_open_stream(
                message, received, self.config
            )

        elif msg_type == "Connected":
            return StreamingTranscriptPrinter.render_connected(
                message, received, self.config
            )

        elif msg_type == "UtteranceEnd":
            return StreamingTranscriptPrinter.render_utterance_end(
                message, received, eot_latency, self.config
            )

        elif msg_type == "StartOfTurn":
            return StreamingTranscriptPrinter.render_start_of_turn(
                message, received, self.config
            )

        elif msg_type == "Results":
            # Record for TTFT and update frequency tracking
            if self.response_metrics_aggregator is not None and received is not None:
                self.response_metrics_aggregator.record_transcript_message(
                    message, received
                )
            return StreamingTranscriptPrinter.render_results(
                message,
                received,
                latency,
                eot_latency,
                self.config,
                self.speaker_aggregator,
                self.use_rich_markup,
            )

        elif msg_type == "TurnInfo":
            # Record for TTFT and update frequency tracking
            if self.response_metrics_aggregator is not None and received is not None:
                self.response_metrics_aggregator.record_transcript_message(
                    message, received
                )
            return StreamingTranscriptPrinter.render_turn_info(
                message,
                received,
                latency,
                eot_latency,
                self.config,
                self.speaker_aggregator,
                self.use_rich_markup,
            )

        elif msg_type == "Metadata":
            return StreamingTranscriptPrinter.render_metadata(
                message, received, self.config
            )

        return None


def fix_input_formatting(response: dict):
    """
    Reformat the transcript field to have a "results" key, if needed.
    """
    # This only works for batch/prerecorded requests right now.
    if "body" in response:
        response["results"] = response.pop("body")
    elif "transcript" in response:
        assert "body" in response["transcript"]
        results = response["transcript"]["body"]
        del response["transcript"]
        response["results"] = results


def remove_text_in_brackets(text: str, remove_newlines: bool = False) -> str:
    """
    Remove all text in square brackets, including the brackets.
    """
    lines = text.split("\n")
    # Ignore UtteranceEnd messages
    lines = [line for line in lines if "[UtteranceEnd]" not in line]
    text = "\n".join(lines)
    while "[" in text:
        start = text.index("[")
        end = text.index("]", start)
        text = text[:start] + text[end + 1 :]
    while "  " in text:
        text = text.replace("  ", " ")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = line.strip(":")
        line = line.strip()
        if line:
            lines.append(line)
    if remove_newlines:
        text = " ".join(lines)
    else:
        text = "\n".join(lines)
    text = text.strip()
    return text


# ============================================================================
# CLI
# ============================================================================


# For boolean flags, see https://click.palletsprojects.com/en/stable/options/#boolean-flags
@click.group(invoke_without_command=True)
@click.option(
    "--filename",
    "-f",
    required=True,
    help="the JSON file to print, including support for streamed JSON messages",
)
@click.option(
    "--only-transcript",
    default=False,
    is_flag=True,
    help="only print the transcript",
)
@click.option(
    "--print-channels/--skip-channels", default=False, help="display the channels"
)
@click.option(
    "--print-speakers/--skip-speakers", default=False, help="display the speaker labels"
)
@click.option(
    "--print-interim/--skip-interim",
    default=False,
    help="display interim results",
)
@click.option(
    "--print-received/--skip-received",
    default=False,
    help="display the received time of a streamed message, if available",
)
@click.option(
    "--print-latency/--skip-latency",
    default=False,
    help="display latency metrics: TTFT, update frequency, message latency, and EOT latency",
)
@click.option(
    "--print-entities/--skip-entities", default=False, help="display entities"
)
@click.option(
    "--colorize/--no-colorize",
    default=False,
    help="colorize the words based on confidence",
)
def main(
    filename: str,
    only_transcript: bool,
    print_channels: bool,
    print_speakers: bool,
    print_interim: bool,
    print_received: bool,
    print_latency: bool,
    print_entities: bool,
    colorize: bool,
):
    # Latency requires interim results to be printed
    if print_latency:
        print_interim = True

    # Create config object
    config = DisplayConfig(
        print_channels=print_channels,
        print_speakers=print_speakers,
        print_interim=print_interim,
        print_received=print_received,
        print_latency=print_latency,
        print_entities=print_entities,
        colorize=colorize,
        only_transcript=only_transcript,
    )

    with open(filename) as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "body" in data or "transcript" in data:
            fix_input_formatting(data)
        assert "results" in data or "err_code" in data
        transcript = formatted_batch_transcript(data, config)
        if not only_transcript:
            print(transcript)
        else:
            print(remove_text_in_brackets(transcript))
    elif isinstance(data, list):
        transcript = StreamingTranscriptPrinter.get_transcript(data, config)
        if not only_transcript:
            print(transcript)
        else:
            print(remove_text_in_brackets(transcript))
    else:
        raise ValueError("impossible")


if __name__ == "__main__":
    main()
