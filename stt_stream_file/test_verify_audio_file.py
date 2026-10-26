import shutil
import subprocess

import pytest

from stream_audio_file import AudioMetadataMismatch, VerifyAudioFile

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required to generate fixtures"
)

DURATION = 2


def _ffmpeg(path, rate, channels, *output_args, source="anoisesrc=c=pink:a=0.3"):
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"{source}:d={DURATION}",
            "-ar", str(rate), "-ac", str(channels),
            *output_args, str(path),
        ],
        check=True,
    )
    return str(path)


@pytest.fixture(scope="session")
def audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    return {
        "wav_s16_16k_mono": _ffmpeg(d / "s16.wav", 16000, 1),
        "wav_s16_16k_stereo": _ffmpeg(d / "stereo.wav", 16000, 2),
        "wav_mulaw_8k_mono": _ffmpeg(d / "mulaw.wav", 8000, 1, "-c:a", "pcm_mulaw"),
        "wav_f32_16k_mono": _ffmpeg(d / "f32.wav", 16000, 1, "-c:a", "pcm_f32le"),
        "flac_16k_mono": _ffmpeg(d / "a.flac", 16000, 1),
        "mp3_44k_mono": _ffmpeg(d / "a.mp3", 44100, 1),
        "ogg_opus_48k_stereo": _ffmpeg(d / "a.ogg", 48000, 2, "-c:a", "libopus"),
        "raw_s16_16k_mono": _ffmpeg(d / "a.raw", 16000, 1, "-f", "s16le"),
        "raw_mulaw_8k_mono": _ffmpeg(d / "a.mulaw", 8000, 1, "-f", "mulaw"),
    }


@pytest.mark.parametrize(
    "name, channels, rate",
    [
        ("wav_s16_16k_mono", 1, 16000),
        ("wav_s16_16k_stereo", 2, 16000),
        ("flac_16k_mono", 1, 16000),
        ("mp3_44k_mono", 1, 44100),
        ("ogg_opus_48k_stereo", 2, 48000),
    ],
)
def test_container_without_url_params_is_detected(audio, name, channels, rate):
    got_channels, _, got_rate, duration = VerifyAudioFile.analyze(audio[name])
    assert (got_channels, got_rate) == (channels, rate)
    assert duration == pytest.approx(DURATION, abs=0.1)


@pytest.mark.parametrize(
    "name, encoding, channels, rate",
    [
        ("wav_s16_16k_mono", "linear16", 1, 16000),
        ("wav_mulaw_8k_mono", "mulaw", 1, 8000),
        ("wav_f32_16k_mono", "linear32", 1, 16000),
        ("raw_s16_16k_mono", "linear16", 1, 16000),
        ("raw_mulaw_8k_mono", "mulaw", 1, 8000),
        ("flac_16k_mono", "flac", 1, 16000),
        ("ogg_opus_48k_stereo", "ogg-opus", 2, 48000),
    ],
)
def test_matching_url_params_are_accepted(audio, name, encoding, channels, rate):
    got_channels, _, got_rate, duration = VerifyAudioFile.analyze(
        audio[name], encoding, channels, rate
    )
    assert (got_channels, got_rate) == (channels, rate)
    assert duration == pytest.approx(DURATION, abs=0.1)


@pytest.mark.parametrize(
    "name, encoding, channels, rate, message",
    [
        ("wav_s16_16k_mono", "linear16", 1, 8000, "sample rate"),
        ("wav_s16_16k_stereo", "linear16", 1, 16000, "channels"),
        ("ogg_opus_48k_stereo", "linear16", 1, 16000, "channels"),
        ("mp3_44k_mono", "linear16", 1, 16000, "sample rate"),
        ("flac_16k_mono", "linear16", 1, 16000, "got flac"),
        ("mp3_44k_mono", "linear16", 1, 44100, "got mp3"),
        ("wav_s16_16k_mono", "mulaw", 1, 16000, "got pcm_s16le"),
        ("wav_mulaw_8k_mono", "linear16", 1, 8000, "got pcm_mulaw"),
    ],
)
def test_mismatched_container_raises(audio, name, encoding, channels, rate, message):
    with pytest.raises(AudioMetadataMismatch, match=message):
        VerifyAudioFile.analyze(audio[name], encoding, channels, rate)


def test_raw_audio_without_url_params_is_rejected(audio):
    with pytest.raises(ValueError) as exc_info:
        VerifyAudioFile.analyze(audio["raw_s16_16k_mono"])
    assert not isinstance(exc_info.value, AudioMetadataMismatch)


def test_defaulted_channels_are_explained(audio):
    with pytest.raises(AudioMetadataMismatch, match="`channels` defaults to 1"):
        VerifyAudioFile.analyze(audio["wav_s16_16k_stereo"], "linear16", 1, 16000)