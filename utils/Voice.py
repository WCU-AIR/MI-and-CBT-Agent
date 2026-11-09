
import io
import os
import re
import wave

import requests
from elevenlabs import ElevenLabs


VOICES = {
    "kokoro": {
        "male": "am_michael",
        "female": "af_heart",
    },
    "qwen": {
        "male": "ryan",
        "female": "vivian",
    },
    "elevenlabs": {
        "male": "iP95p4xoKVk53GoZ742B",
        "female": "cgSgspJ2msm6clMCkdW9",
    },
}


class Voice:
    def __init__(self, case_study: str, api_key: str = ""):
        self.provider = os.getenv(
            "TTS_PROVIDER", "local"
        ).lower()

        self.model = os.getenv(
            "TTS_MODEL", "kokoro"
        ).lower()

        self.local_url = os.getenv(
            "LOCAL_TTS_URL", "http://127.0.0.1:8880"
        ).rstrip("/")

        self.case_study = case_study

        if self.provider not in {"local", "elevenlabs"}:
            raise ValueError("Unsupported TTS provider")

        if self.provider == "local":
            if self.model not in {"kokoro", "qwen"}:
                raise ValueError("Unsupported local TTS model")
        else:
            self.model = "elevenlabs"

        self.client = (
            ElevenLabs(api_key=api_key)
            if self.provider == "elevenlabs"
            else None
        )

        self.patient_sex = self._get_patient_sex()
        self.patient_voice = VOICES[self.model][
            self.patient_sex
        ]

        opposite = (
            "female" if self.patient_sex == "male"
            else "male"
        )
        self.evaluator_voice = VOICES[self.model][
            opposite
        ]

    def _get_patient_sex(self) -> str:
        match = re.search(
            r"Sex:\s*(Male|Female)",
            self.case_study,
            re.IGNORECASE,
        )
        return (
            match.group(1).lower()
            if match else "female"
        )

    def text_to_speech(
        self, text: str, speaker: str = "patient"
    ) -> tuple[bytes, str]:

        voice = (
            self.patient_voice
            if speaker == "patient"
            else self.evaluator_voice
        )

        if self.provider == "elevenlabs":
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice,
                model_id="eleven_monolingual_v1",
            )
            return b"".join(audio_stream), "audio/mpeg"

        response = requests.post(
            f"{self.local_url}/v1/text-to-speech/"
            f"{voice}/stream",
            json={
                "text": text,
                "model_id": self.model,
                "voice_settings": {"speed": 1.0},
            },
            timeout=180,
        )
        response.raise_for_status()

        # local-tts returns raw PCM, not a WAV file.
        # Wrap it in a WAV container for browser playback.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(response.content)

        return buffer.getvalue(), "audio/wav"
