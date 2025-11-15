import os
from dotenv import load_dotenv
from io import BytesIO
from elevenlabs.client import ElevenLabs
from utils.api_key_loader import load_elevenlabs_api_key


class STT:
    """
    A class to handle voice interactions, now using ElevenLabs for Speech-to-Text.
    """

    def __init__(self):
        """
        Initializes the class, loading the ElevenLabs API key and creating a client.
        """

        elevenlabs_api_key = load_elevenlabs_api_key()
        if not elevenlabs_api_key:
            raise ValueError("ElevenLabs API key not found in .clinician_env file.")

        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)

    def speech_to_text(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio bytes into text using the ElevenLabs Scribe v1 model.

        This method sends the audio data directly from memory to the ElevenLabs API.

        Args:
            audio_bytes: The audio data in bytes recorded from the microphone.

        Returns:
            The transcribed text as a string.
        """
        try:
            audio_data = BytesIO(audio_bytes)

            transcription = self.elevenlabs_client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                language_code="eng",  # English language
            )

            return transcription.text

        except Exception as e:
            print(f"Error during ElevenLabs transcription: {e}")
            return ""
