import os
import json
import tempfile
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI

# Load .env from the same folder as this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

app = FastAPI(title="REVE Speaking AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def azure_is_configured() -> bool:
    return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)


def openai_is_configured() -> bool:
    return bool(OPENAI_API_KEY)


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def safe_json(text: str) -> dict:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

    return {}


def azure_pronunciation_assessment(
    audio_path: str,
    reference_text: str,
    language: str,
) -> dict:
    if not azure_is_configured():
        raise RuntimeError("Azure Speech is not configured.")

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )

    # Pronunciation Assessment is intended for speech/language learning.
    # The current backend expects an English reference sentence.
    speech_config.speech_recognition_language = language or "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text or "",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )

    # Prosody is supported on SDK versions that expose this property.
    try:
        pronunciation_config.enable_prosody_assessment()
    except Exception:
        pass

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        cancellation = getattr(result, "cancellation_details", None)
        reason = getattr(cancellation, "reason", "Speech recognition failed")
        details = getattr(cancellation, "error_details", "")
        raise RuntimeError(f"{reason}. {details}".strip())

    transcript = result.text or ""

    raw_json = {}
    try:
        raw_json = json.loads(
            result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult,
                "{}",
            )
        )
    except Exception:
        raw_json = {}

    pronunciation_result = {}
    try:
        pronunciation_result = json.loads(
            result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult,
                "{}",
            )
        ).get("NBest", [{}])[0].get("PronunciationAssessment", {})
    except Exception:
        pronunciation_result = {}

    # Azure's pronunciation values may also be available in the result JSON.
    accuracy = clamp_score(pronunciation_result.get("AccuracyScore", 0))
    fluency = clamp_score(pronunciation_result.get("FluencyScore", 0))
    completeness = clamp_score(pronunciation_result.get("CompletenessScore", 0))
    prosody = clamp_score(pronunciation_result.get("ProsodyScore", 0))

    pronunciation_score = accuracy

    # Keep pronunciation meaningful even when prosody is unavailable.
    if prosody > 0:
        pronunciation_score = round((accuracy * 0.8) + (prosody * 0.2))

    return {
        "transcript": transcript,
        "pronunciation_score": clamp_score(pronunciation_score),
        "accuracy_score": accuracy,
        "fluency_score": fluency,
        "completeness_score": completeness,
        "prosody_score": prosody,
        "raw": raw_json,
    }


def openai_language_evaluation(
    reference_text: str,
    transcript: str,
) -> dict:
    if not openai_is_configured():
        return {
            "corrected_sentence": "",
            "grammar_score": 0,
            "vocabulary_score": 0,
            "feedback": "OpenAI evaluation is not configured yet.",
            "next_practice": "Configure OPENAI_API_KEY in .env.",
            "mistakes": [],
        }

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """
You are an English speaking coach.
Evaluate the learner's spoken transcript against the reference sentence.

Return ONLY valid JSON with exactly these keys:
{
  "corrected_sentence": "string",
  "grammar_score": 0,
  "vocabulary_score": 0,
  "feedback": "string",
  "next_practice": "string",
  "mistakes": [
    {
      "original": "string",
      "correction": "string",
      "explanation": "string"
    }
  ]
}

Scores must be integers from 0 to 100.
Grammar score measures grammatical correctness.
Vocabulary score measures appropriateness, variety and accuracy of vocabulary.
Do not invent pronunciation scores; pronunciation is handled separately.
Keep feedback concise and useful for a language learner.
"""

    user_prompt = (
        f"Reference sentence:\n{reference_text}\n\n"
        f"Learner transcript:\n{transcript}\n"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    data = safe_json(content)

    return {
        "corrected_sentence": str(data.get("corrected_sentence", "")),
        "grammar_score": clamp_score(data.get("grammar_score", 0)),
        "vocabulary_score": clamp_score(data.get("vocabulary_score", 0)),
        "feedback": str(data.get("feedback", "")),
        "next_practice": str(data.get("next_practice", "")),
        "mistakes": data.get("mistakes", [])
        if isinstance(data.get("mistakes", []), list)
        else [],
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "azure_configured": azure_is_configured(),
        "openai_configured": openai_is_configured(),
        "azure_region": AZURE_SPEECH_REGION if azure_is_configured() else "",
        "openai_model": OPENAI_MODEL if openai_is_configured() else "",
    }


@app.post("/evaluate-speaking")
async def evaluate_speaking(
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    language: str = Form("en-US"),
):
    if not reference_text.strip():
        raise HTTPException(
            status_code=400,
            detail="reference_text is required.",
        )

    if not azure_is_configured():
        raise HTTPException(
            status_code=503,
            detail="Azure Speech is not configured. Check .env.",
        )

    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    temp_path = ""

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        azure_result = azure_pronunciation_assessment(
            audio_path=temp_path,
            reference_text=reference_text.strip(),
            language=language.strip() or "en-US",
        )

        transcript = azure_result["transcript"]

        ai_result = openai_language_evaluation(
            reference_text=reference_text.strip(),
            transcript=transcript,
        )

        pronunciation = clamp_score(
            azure_result.get("pronunciation_score", 0)
        )
        grammar = clamp_score(ai_result.get("grammar_score", 0))
        vocabulary = clamp_score(ai_result.get("vocabulary_score", 0))
        fluency = clamp_score(azure_result.get("fluency_score", 0))

        # Genuine overall score from actual Azure pronunciation/fluency
        # plus OpenAI grammar/vocabulary evaluation.
        overall = round(
            pronunciation * 0.45
            + grammar * 0.25
            + vocabulary * 0.20
            + fluency * 0.10
        )

        return {
            "ok": True,
            "transcript": transcript,
            "reference_text": reference_text.strip(),
            "corrected_sentence": ai_result.get("corrected_sentence", ""),
            "grammar_score": grammar,
            "vocabulary_score": vocabulary,
            "fluency_score": fluency,
            "pronunciation_score": pronunciation,
            "overall_score": clamp_score(overall),
            "accuracy_score": azure_result.get("accuracy_score", 0),
            "completeness_score": azure_result.get("completeness_score", 0),
            "prosody_score": azure_result.get("prosody_score", 0),
            "mistakes": ai_result.get("mistakes", []),
            "feedback": ai_result.get("feedback", ""),
            "next_practice": ai_result.get("next_practice", ""),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speaking evaluation failed: {str(exc)}",
        )
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reve_speaking_ai_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
