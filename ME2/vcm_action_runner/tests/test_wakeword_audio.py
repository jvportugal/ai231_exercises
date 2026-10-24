import io
import unittest
import wave
from unittest import mock

import numpy as np

from vcm_action_runner import wakeword as W


class _OutputStream:
    def __init__(self, owner, **settings):
        self.owner = owner
        self.settings = settings

    def start(self):
        self.owner.started += 1

    def write(self, data):
        self.owner.data = np.asarray(data)

    def stop(self):
        self.owner.stopped += 1

    def close(self):
        self.owner.closed += 1


class _SoundDevice:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.closed = 0
        self.data = None

    def query_devices(self, _device, kind):
        assert kind == "output"
        return {"name": "Pi PCM", "default_samplerate": 48000,
                "max_output_channels": 2}

    def OutputStream(self, **settings):
        if settings["dtype"] != "int16":
            raise RuntimeError("sample format not supported")
        return _OutputStream(self, **settings)


class _Microphone:
    def __init__(self, blocks):
        self.blocks = iter(blocks)

    def read(self):
        return next(self.blocks)


class WakewordAudioTests(unittest.TestCase):
    def test_chime_uses_native_int16_stereo_output(self):
        sd = _SoundDevice()
        audio = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
        result = W._play_audio_compatible(sd, audio)
        self.assertEqual(result["rate"], 48000)
        self.assertEqual(result["dtype"], "int16")
        self.assertEqual(result["channels"], 2)
        self.assertEqual(sd.data.dtype, np.int16)
        self.assertEqual(sd.data.shape, (4800, 2))
        self.assertEqual((sd.started, sd.stopped, sd.closed), (1, 1, 1))

    @mock.patch.object(W.subprocess, "run")
    @mock.patch.object(W.shutil, "which", return_value="/usr/bin/aplay")
    def test_aplay_backend_sends_native_alsa_wav(self, _which, run):
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        audio = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)

        result = W._play_audio_aplay(audio, device="plughw:CARD=Headphones,DEV=0")

        command = run.call_args.args[0]
        self.assertEqual(command,
                         ["/usr/bin/aplay", "-q", "-D",
                          "plughw:CARD=Headphones,DEV=0"])
        payload = run.call_args.kwargs["input"]
        with wave.open(io.BytesIO(payload), "rb") as wav:
            self.assertEqual(wav.getframerate(), 48000)
            self.assertEqual(wav.getnchannels(), 2)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 4800)
        self.assertEqual(result["backend"], "aplay")

    def test_short_speech_level_preserves_quick_40_ms_word(self):
        block = np.zeros(W.BLOCK_SAMPLES, np.float32)
        block[:W.SAMPLE_RATE * 40 // 1000] = 0.012
        self.assertLess(W.block_rms(block), 0.008)
        self.assertGreater(W.short_speech_level(block), 0.008)

    def test_short_speech_level_rejects_isolated_click(self):
        block = np.zeros(W.BLOCK_SAMPLES, np.float32)
        block[10] = 0.9
        self.assertLess(W.short_speech_level(block), 0.008)

    def test_one_block_command_finishes_without_hard_timeout(self):
        speech = np.zeros(W.BLOCK_SAMPLES, np.float32)
        speech[:W.SAMPLE_RATE * 40 // 1000] = 0.012
        silence = np.zeros(W.BLOCK_SAMPLES, np.float32)

        session = W.WakeWordSession.__new__(W.WakeWordSession)
        session.guard_blocks = 0
        session.silence_blocks = 3
        session.command_timeout = 6.0
        session.speech_start_timeout = 4.0
        session.speech_gate = W.AdaptiveSpeechGate(minimum=0.008)
        session.microphone = _Microphone([speech, silence, silence, silence])
        session.last_timing = {}

        audio = session._capture_command(None)
        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), 4 * W.BLOCK_SAMPLES)
        self.assertEqual(session.last_timing["command_endpoint_ms"], 300.0)

    def test_post_chime_block_can_start_an_immediate_short_command(self):
        speech = np.zeros(W.BLOCK_SAMPLES, np.float32)
        speech[:W.SAMPLE_RATE * 40 // 1000] = 0.012
        silence = np.zeros(W.BLOCK_SAMPLES, np.float32)

        session = W.WakeWordSession.__new__(W.WakeWordSession)
        session.guard_blocks = 0
        session.silence_blocks = 3
        session.command_timeout = 6.0
        session.speech_start_timeout = 4.0
        session.speech_gate = W.AdaptiveSpeechGate(minimum=0.008)
        session.microphone = _Microphone([silence, silence, silence])
        session.last_timing = {}

        audio = session._capture_command(None, initial_blocks=[speech])

        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), 4 * W.BLOCK_SAMPLES)
        self.assertLess(session.last_timing["command_speech_wait_ms"], 50.0)

    def test_wake_callback_runs_before_command_capture(self):
        speech = np.full(W.BLOCK_SAMPLES, 0.02, np.float32)
        silence = np.zeros(W.BLOCK_SAMPLES, np.float32)
        events = []

        session = W.WakeWordSession.__new__(W.WakeWordSession)
        session.last_timing = {}
        session.microphone = _Microphone([speech, silence])
        session.speech_gate = W.AdaptiveSpeechGate(minimum=0.008)
        session.wake_silence_blocks = 1
        session.wake_max_blocks = 20
        session.threshold = 0.5
        session.detector = mock.Mock()
        session.detector.score.return_value = 0.9
        session.on_wake = lambda: events.append("wake")
        session._play_chime = mock.Mock(
            side_effect=lambda: events.append("chime") or 0.0)
        session._capture_command = mock.Mock(
            side_effect=lambda *_args, **_kwargs: events.append("capture") or speech)

        audio = session.wait_for_command()

        self.assertIs(audio, speech)
        self.assertEqual(events, ["wake", "chime", "capture"])


if __name__ == "__main__":
    unittest.main()
