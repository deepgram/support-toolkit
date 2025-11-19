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
    print_delay: bool = False
    print_entities: bool = False
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
    delay: Optional[float] = None

    def format(self) -> str:
        """Build the final formatted line"""
        parts = []

        # Add received timestamp if configured
        if self.config.print_received and self.received is not None:
            if self.config.print_delay:
                received_str = f"[{self.received.strftime('%H:%M:%S.%f')} ({self.delay:.4f}s since EOS)]"
            else:
                received_str = f"[{self.received.strftime('%H:%M:%S.%f')}]"
            parts.append(received_str)

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
    delay: float | None,
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
        delay=delay,
        config=config,
    )

    return line_builder.format()


# ============================================================================
# Delay Tracking
# ============================================================================


class DelayAggregator:
    def __init__(self, start: datetime.datetime) -> None:
        self.start = start
        self.last_message = start
        self.last_is_final = start
        self.last_speech_final = start
        self.last_utterance_end = start
        self.last_word_end = datetime.timedelta(seconds=0)

    @staticmethod
    def calculate_start_from_messages(messages: list[dict]) -> datetime.datetime | None:
        # Look for an OpenStream message, if it exists.
        for message in messages:
            if message.get("type") == "OpenStream":
                return datetime.datetime.fromisoformat(message["received"])

        # If there is no OpenStream message, we can try to estimate it.
        approx_stream_open_time = None
        for message in messages:
            if "received" not in message:
                continue
            received = datetime.datetime.fromisoformat(message["received"])
            # Since this is a streaming request, we should assume the stream was opened when the audio started.
            # This may not be accurate because the user can send the audio faster or slower than real-time,
            # but it should work for our needs.
            if message.get("type") == "Results":
                approx_stream_open_time = received - (
                    datetime.timedelta(seconds=message["start"])
                    + datetime.timedelta(seconds=message["duration"])
                )
            elif message.get("type") == "TurnInfo":
                # For Flux, use audio_window_end
                approx_stream_open_time = received - datetime.timedelta(
                    seconds=message.get("audio_window_end", 0)
                )
            # Break after the first message with a received time
            if approx_stream_open_time:
                break
        return approx_stream_open_time

    def add_message(self, message: dict, received: datetime.datetime):
        self.last_message = received
        if message.get("type") == "UtteranceEnd":
            self.last_utterance_end = received
        elif message.get("type") == "Results":
            if message.get("is_final", False):
                self.last_is_final = received
            if message.get("speech_final", False):
                self.last_speech_final = received
            for word in message["channel"]["alternatives"][0]["words"]:
                if "end" in word and word["end"] > 0:
                    self.last_word_end = datetime.timedelta(seconds=word["end"])
        elif message.get("type") == "TurnInfo":
            # For Flux, handle EndOfTurn events
            if message.get("event") == "EndOfTurn":
                self.last_is_final = received
                self.last_speech_final = received
            # Update last word end time from audio window
            if message.get("audio_window_end", 0) > 0:
                self.last_word_end = datetime.timedelta(
                    seconds=message["audio_window_end"]
                )

    def get_message_delay(self, received: datetime.datetime) -> datetime.timedelta:
        return received - self.last_message

    def get_is_final_delay(self, received: datetime.datetime) -> datetime.timedelta:
        return received - self.last_is_final

    def get_speech_final_delay(self, received: datetime.datetime) -> datetime.timedelta:
        return received - self.last_speech_final

    def get_finalized_delay(self, received: datetime.datetime) -> datetime.timedelta:
        last = max(self.last_speech_final, self.last_utterance_end)
        return (received - last) - self.last_word_end

    def get_eos_delay(self, received: datetime.datetime) -> datetime.timedelta:
        """
        Returns the delay between the received time and when the last word was spoken.
        This is a time-since-end-of-speech calculation.
        """
        return (received - self.start) - self.last_word_end


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
            delay=None,
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
        delay: float | None,
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
        delay: float | None,
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
        delay: float | None,
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
        delay: float | None,
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
            delay=delay,
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
        delay: float | None,
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
            if config.print_delay:
                received_str = (
                    f" [{received.strftime('%H:%M:%S.%f')} ({delay:.4f}s since EOS)]"
                )
            else:
                received_str = f"[{received.strftime('%H:%M:%S.%f')}]"
        else:
            received_str = ""
        line = f"{received_str} {channel_str} [UtteranceEnd]: last_word_end={last_word_end}"
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
        delay: float | None,
        config: DisplayConfig,
    ) -> str:
        channel = message["channel"][0]
        if config.print_channels:
            channel_str = f"[Channel {channel}]"
        else:
            channel_str = ""
        if config.print_received and received is not None:
            if config.print_delay:
                received_str = (
                    f" [{received.strftime('%H:%M:%S.%f')} ({delay:.4f}s since EOS)]"
                )
            else:
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
        delay: float | None,
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
            delay=delay,
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
        stream_open_time = DelayAggregator.calculate_start_from_messages(messages)
        if stream_open_time is not None:
            delay_aggregator = DelayAggregator(start=stream_open_time)
        else:
            delay_aggregator = None
            config.print_delay = False

        single_line_transcripts: list[str] = []
        for message in messages:
            # Update the DelayAggregator and `delay`
            delay = None
            received_iso = message.get("received", None)
            if received_iso is None:
                received = None
            else:
                received = datetime.datetime.fromisoformat(received_iso)
                if delay_aggregator is not None:
                    delay = delay_aggregator.get_eos_delay(received).total_seconds()
                    delay_aggregator.add_message(message, received)

            # Handle UtteranceEnd messages separately because they have a different type.
            if message.get("type") == "UtteranceEnd":
                line = cls.render_utterance_end(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                )
                single_line_transcripts.append(line)

            # Print the OpenStream message if it exists.
            elif message.get("type") == "OpenStream":
                line = cls.render_open_stream(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                )
                single_line_transcripts.append(line)
            # Print the Connected message (Flux)
            elif message.get("type") == "Connected":
                line = cls.render_connected(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                )
                single_line_transcripts.append(line)

            # Print the Metadata/CloseStream message if it exists.
            elif message.get("type") == "Metadata":
                line = cls.render_metadata(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                )
                single_line_transcripts.append(line)

            elif message.get("type") == "Results":
                line = cls.render_results(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                    speaker_aggregator=speaker_aggregator,
                )
                if line is not None:
                    single_line_transcripts.append(line)

            elif message.get("type") == "StartOfTurn":
                line = cls.render_start_of_turn(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                )
                single_line_transcripts.append(line)

            elif message.get("type") == "TurnInfo":
                line = cls.render_turn_info(
                    message=message,
                    received=received,
                    delay=delay,
                    config=config,
                    speaker_aggregator=speaker_aggregator,
                )
                if line is not None:
                    single_line_transcripts.append(line)
            else:
                print("Skipping message of type:", message.get("type"))

        transcript = "\n".join(single_line_transcripts)
        return transcript


class StreamingFormatter:
    """Stateful formatter for processing messages one at a time"""

    def __init__(
        self,
        config: DisplayConfig,
        stream_open_time: datetime.datetime | None = None,
        use_rich_markup: bool = False,
    ):
        self.config = config
        self.speaker_aggregator = SpeakerAggregator()
        self.use_rich_markup = use_rich_markup

        if stream_open_time:
            self.delay_aggregator = DelayAggregator(start=stream_open_time)
            self.config.print_delay = True
        else:
            self.delay_aggregator = None
            self.config.print_delay = False

    def format_message(self, message: dict) -> str | None:
        """Format a single message and return the formatted string"""

        # Update delay tracking
        received = None
        delay = None
        received_iso = message.get("received", None)
        if received_iso is not None:
            received = datetime.datetime.fromisoformat(received_iso)
            if self.delay_aggregator is not None:
                delay = self.delay_aggregator.get_eos_delay(received).total_seconds()
                self.delay_aggregator.add_message(message, received)

        # Handle different message types
        msg_type = message.get("type")

        if msg_type == "OpenStream":
            # Update delay aggregator start time if not set
            if self.delay_aggregator is None and received:
                self.delay_aggregator = DelayAggregator(start=received)
            return StreamingTranscriptPrinter.render_open_stream(
                message, received, delay, self.config
            )

        elif msg_type == "Connected":
            if self.delay_aggregator is None and received:
                self.delay_aggregator = DelayAggregator(start=received)
            return StreamingTranscriptPrinter.render_connected(
                message, received, delay, self.config
            )

        elif msg_type == "UtteranceEnd":
            return StreamingTranscriptPrinter.render_utterance_end(
                message, received, delay, self.config
            )

        elif msg_type == "StartOfTurn":
            return StreamingTranscriptPrinter.render_start_of_turn(
                message, received, delay, self.config
            )

        elif msg_type == "Results":
            return StreamingTranscriptPrinter.render_results(
                message,
                received,
                delay,
                self.config,
                self.speaker_aggregator,
                self.use_rich_markup,
            )

        elif msg_type == "TurnInfo":
            return StreamingTranscriptPrinter.render_turn_info(
                message,
                received,
                delay,
                self.config,
                self.speaker_aggregator,
                self.use_rich_markup,
            )

        elif msg_type == "Metadata":
            return StreamingTranscriptPrinter.render_metadata(
                message, received, delay, self.config
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
    "--print-delay/--skip-delay", default=False, help="display the EOS latency"
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
    print_delay: bool,
    print_entities: bool,
    colorize: bool,
):
    if print_delay:
        print_received = True

    # Create config object
    config = DisplayConfig(
        print_channels=print_channels,
        print_speakers=print_speakers,
        print_interim=print_interim,
        print_received=print_received,
        print_delay=print_delay,
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
