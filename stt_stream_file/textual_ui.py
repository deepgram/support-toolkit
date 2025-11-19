import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Static
from textual.containers import Container, Vertical, ScrollableContainer
from textual.reactive import reactive

from stream_audio_file import stream_audio
from print_transcript import DisplayConfig, StreamingFormatter


class TranscriptionApp(App):
    """A Textual app for real-time audio transcription."""

    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 100%;
        height: 100%;
        border: solid $primary;
    }

    #info-panel {
        height: auto;
        background: $surface;
        padding: 1;
        border-bottom: solid $primary;
    }

    #transcript-container {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }

    #status {
        height: 3;
        background: $surface;
        padding: 1;
        border-top: solid $primary;
    }

    .info-line {
        margin: 0 1;
    }
    """

    TITLE = "Real-Time Transcription"

    status_text = reactive("Ready")

    def __init__(
        self,
        audio_file: str | None,
        url: str,
        encoding: str | None,
        channels: int | None,
        sample_rate: int | None,
        verbose: int,
        config: DisplayConfig,
    ):
        super().__init__()
        self.audio_file = audio_file
        self.url = url
        self.encoding = encoding
        self.channels = channels
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.config = config
        self.formatter: StreamingFormatter | None = None
        self.stream_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Vertical(id="main-container"):
            with Container(id="info-panel"):
                yield Static("🎤 Audio Source:", classes="info-line")
                yield Static(
                    f"   {'Microphone (live)' if self.audio_file is None else self.audio_file}",
                    classes="info-line",
                )
                if self.config.print_speakers:
                    yield Static("👤 Showing speakers", classes="info-line")
                if self.config.print_channels:
                    yield Static("📢 Showing channels", classes="info-line")
                if self.config.print_interim:
                    yield Static("⏱️  Showing interim results", classes="info-line")

            with ScrollableContainer(id="transcript-container"):
                yield RichLog(id="transcript", wrap=True, highlight=True, markup=True)

            with Container(id="status"):
                yield Static(self.status_text, id="status-display")

        yield Footer()

    async def on_mount(self) -> None:
        """Start streaming when the app mounts."""
        self.transcript_log = self.query_one("#transcript", RichLog)
        self.status_display = self.query_one("#status-display", Static)

        # Initialize formatter with Rich markup enabled
        self.formatter = StreamingFormatter(self.config, use_rich_markup=True)

        # Start streaming
        self.update_status("Connecting to Deepgram...")
        self.stream_task = asyncio.create_task(self.start_streaming())

    def update_status(self, text: str):
        """Update the status display."""
        self.status_text = text
        self.status_display.update(text)

    async def message_callback(self, message: dict):
        """Called for each message received from Deepgram."""
        if self.formatter is None:
            return

        # Format the message (with Rich markup if colorize is enabled)
        formatted = self.formatter.format_message(message)

        # Update status based on message type
        msg_type = message.get("type")
        if msg_type == "OpenStream" or msg_type == "Connected":
            self.update_status("🟢 Connected - Streaming audio...")
        elif msg_type == "Results":
            if message.get("speech_final"):
                self.update_status("✅ Processing audio...")
            else:
                self.update_status("🎙️ Listening...")
        elif msg_type == "Metadata":
            self.update_status("✅ Transcription complete")

        # Display the formatted message (Rich markup is automatically rendered)
        if formatted:
            self.transcript_log.write(formatted)

    async def start_streaming(self):
        """Start the audio streaming process."""
        try:
            # Use a temporary output file (won't be used in UI mode)
            temp_output = "/tmp/transcript_temp.json"

            await stream_audio(
                output_filename=temp_output,
                filename=self.audio_file,
                url=self.url,
                encoding=self.encoding,
                channels=self.channels,
                sample_rate=self.sample_rate,
                verbose=self.verbose,
                message_callback=self.message_callback,
            )

            self.update_status("✅ Complete - Press Ctrl+C to exit")

        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            self.transcript_log.write(f"\n[bold red]Error:[/bold red] {str(e)}")

    async def action_quit(self):
        """Quit the application."""
        if self.stream_task:
            self.stream_task.cancel()
        await super().action_quit()


async def launch_ui(
    audio_file: str | None,
    url: str,
    encoding: str | None,
    channels: int | None,
    sample_rate: int | None,
    verbose: int,
    print_channels: bool = False,
    print_speakers: bool = True,
    print_interim: bool = True,
    print_received: bool = True,
    print_delay: bool = True,
    print_entities: bool = False,
    colorize: bool = True,
):
    """Launch the Textual UI with the given configuration."""

    config = DisplayConfig(
        print_channels=print_channels,
        print_speakers=print_speakers,
        print_interim=print_interim,
        print_received=print_received,
        print_delay=print_delay,
        print_entities=print_entities,
        colorize=colorize,
        only_transcript=False,
    )

    app = TranscriptionApp(
        audio_file=audio_file,
        url=url,
        encoding=encoding,
        channels=channels,
        sample_rate=sample_rate,
        verbose=verbose,
        config=config,
    )

    await app.run_async()
