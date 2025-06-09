from elevenlabs import ElevenLabs
import os
import  re

MALE_VOICE = "iP95p4xoKVk53GoZ742B"     #Eleven Labs Default Voice: Chris
FEMALE_VOICE = "cgSgspJ2msm6clMCkdW9"   #Eleven Labs Default Voice: Jessica


# Initialize
api_key = os.getenv("ELEVEN_API_KEY")
if not api_key:
    raise ValueError("ELEVEN_API_KEY not set in environment variables")

client = ElevenLabs(api_key=api_key)  # Now it works!

def get_patient_sex(case_study: str) -> str:
    try:

        # Try to extract from structured section first
        match = re.search(r"Sex:\s*(Male|Female)", case_study, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()

        return "Unknown"

    except Exception as e:
        print(f"Error reading file: {e}")
        return "Unknown"

def set_patient_voice(case_study: str) -> str:
    sex = get_patient_sex(case_study)

    if sex == "Male":
        voice = MALE_VOICE
        print("Male Voice Selected")
    elif sex == "Female":
        voice = FEMALE_VOICE
        print("Female Voice Selected")
    else:
        voice = FEMALE_VOICE
        print("Unknown Voice Detected. Defaulting to female voice")

    return voice

def text_to_speech(text: str, voice: str) -> bytes:
    #Remove text inside parentheses (We don't want to read that part aloud)
    text = re.sub(r'^\s*\([^)]*\)\s*', '', text)

    # Convert text to speech and return raw MP3 bytes
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice,
        model_id="eleven_monolingual_v1",
        voice_settings={
            "stability": 0.25,          # Purpose: Controls how much variation or emotional expressiveness is allowed in the generated speech.
                                        # Lower Value (0.25) = More expressive, emotional, and dynamic.
            "similarity_boost": 0.6     # Purpose: Controls how closely the output matches the original voice's training data.
                                        # Mid-range value (0.6) gives some flexibility, while still remaining
                                        # consistent across multiple utterances
        }
    )
    audio_bytes = b"".join(chunk for chunk in audio_stream)

    return audio_bytes








