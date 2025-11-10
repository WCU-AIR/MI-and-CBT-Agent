import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Import the new modular LearningBuddy
from LearningBuddy_Agent import LearningBuddy, MI_MODE, STACER_MODE
from LearningBuddy_Agent.graph_state import LBState, PatientContext, SessionMetadata
from utils.api_key_loader import load_openai_api_key



class TranscriptEvaluator:
    """
    Handles batch evaluation of MI transcripts using the new modular LearningBuddy.
    """

    def __init__(self, api_key: str, elevenlabs_api_key: str = "", mode: int = MI_MODE):
        """
        Initialize the evaluator.

        Args:
            api_key: OpenAI API key
            elevenlabs_api_key: ElevenLabs API key (optional)
            mode: Evaluation mode (MI_MODE or STACER_MODE)
        """
        self.buddy = LearningBuddy(
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            mode=mode
        )
        self.mode = mode
        logging.info(f"TranscriptEvaluator initialized in mode {mode}")

    def _build_state_from_transcript(self, transcript_rows: pd.DataFrame) -> LBState:
        """
        Build an LBState instance from a motivational interviewing transcript.

        Args:
            transcript_rows: DataFrame containing transcript data

        Returns:
            LBState instance ready for evaluation
        """
        required_columns = {
            "mi_quality",
            "transcript_id",
            "video_title",
            "video_url",
            "topic",
            "utterance_id",
            "interlocutor",
            "timestamp",
            "utterance_text",
        }
        missing_columns = required_columns.difference(transcript_rows.columns)
        if missing_columns:
            raise ValueError(
                f"Transcript is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        transcript_rows = transcript_rows.sort_values("timestamp")
        first_row = transcript_rows.iloc[0]

        def _safe_str(value, fallback):
            if pd.isna(value) or value is None:
                return fallback
            value_str = str(value).strip()
            return value_str if value_str else fallback

        # Extract metadata
        topic = _safe_str(first_row.get("topic"), "Unknown topic")
        video_title = _safe_str(first_row.get("video_title"), "Unknown video title")
        transcript_id = _safe_str(first_row.get("transcript_id"), "unknown_transcript")
        mi_quality = _safe_str(first_row.get("mi_quality"), "unspecified quality")

        # Build case study content
        case_study = (
            f"Topic: {topic}\n"
            f"Video Title: {video_title}\n"
            f"Transcript ID: {transcript_id}\n"
            f"Quality Label: {mi_quality}"
        )

        clinical_summary = (
            f"Motivational interviewing session on '{topic}' labelled '{mi_quality}'. "
            f"Source video: {video_title}."
        )

        # Build interview messages from transcript
        interview_messages = []
        for _, row in transcript_rows.iterrows():
            content = _safe_str(row.get("utterance_text"), "")
            if not content:
                continue
            interlocutor = _safe_str(row.get("interlocutor"), "client").lower()
            # Map interlocutor to message type
            if interlocutor == "therapist":
                interview_messages.append(HumanMessage(content=content))
            else:
                interview_messages.append(AIMessage(content=content))

        if not interview_messages:
            raise ValueError("Transcript contains no utterances.")

        # Calculate timestamps
        timestamps = transcript_rows["timestamp"].dropna().astype(str).str.strip()
        start_timestamp = timestamps.iloc[0] if not timestamps.empty else None
        end_timestamp = timestamps.iloc[-1] if not timestamps.empty else None
        start_seconds = self._timestamp_to_seconds(start_timestamp)
        end_seconds = self._timestamp_to_seconds(end_timestamp)
        transcript_duration_minutes = self._duration_minutes_from_seconds(start_seconds, end_seconds)

        # Build session metadata
        session_metadata: SessionMetadata = {
            "session_id": f"eval_{transcript_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "selected_profile": None,
            "transcript_start_seconds": start_seconds,
            "transcript_end_seconds": end_seconds,
            "transcript_duration_minutes": transcript_duration_minutes,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        }

        # Build patient context
        patient_context: PatientContext = {
            "case_study": case_study,
            "diagnosis_summary": "",
            "clinical_summary": clinical_summary,
            "profile_path": None
        }

        # Build complete state
        state: LBState = {
            # Conversational state
            "interview_messages": interview_messages,
            "feedback_messages": [],
            "evaluator_messages": [],

            # Session control
            "current_phase": "evaluate_session",
            "session_active": False,
            "evaluator_active": False,

            # Nested data structures
            "patient_context": patient_context,
            "presentation": None,
            "evaluation": None,
            "metadata": session_metadata,

            # Optional fields
            "cod_stacer": None,
            "formatted_transcript": "",
            "formatted_presentation": "",
            "error": None,
            "folder_name": None,
        }

        return state

    @staticmethod
    def _timestamp_to_seconds(timestamp: Optional[str]) -> Optional[float]:
        """Convert timestamp string (HH:MM:SS) to seconds."""
        if timestamp is None:
            return None
        value = str(timestamp).strip()
        if not value:
            return None
        parts = value.split(':')
        if len(parts) != 3:
            return None
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        except ValueError:
            return None
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return float(total_seconds)

    @staticmethod
    def _duration_minutes_from_seconds(
            start_seconds: Optional[float],
            end_seconds: Optional[float]
    ) -> Optional[float]:
        """Calculate duration in minutes from start and end seconds."""
        if start_seconds is None or end_seconds is None:
            return None
        if end_seconds < start_seconds:
            return None
        return round((end_seconds - start_seconds) / 60.0, 2)

    def run_directory_evaluations(self, csv_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Run the feedback workflow for every transcript CSV in the given directory.

        Args:
            csv_dir: Path to directory containing transcript CSV files

        Returns:
            Dictionary mapping transcript_id to evaluation results
        """
        csv_path = Path(csv_dir).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Directory not found: {csv_path}")
        if not csv_path.is_dir():
            raise NotADirectoryError(f"Provided path is not a directory: {csv_path}")

        results: Dict[str, Dict[str, str]] = {}

        for csv_file in sorted(csv_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:
                logging.exception("Failed to read transcript file %s: %s", csv_file, exc)
                continue

            if df.empty:
                logging.warning("Skipping empty transcript file %s", csv_file)
                continue

            if "transcript_id" not in df.columns:
                logging.error("Skipping %s because it lacks a transcript_id column", csv_file)
                continue

            # Process each transcript in the file
            for transcript_id, transcript_rows in df.groupby("transcript_id"):
                try:
                    # Build state from transcript
                    state = self._build_state_from_transcript(transcript_rows)
                except ValueError as exc:
                    logging.error(
                        "Skipping transcript %s in %s: %s",
                        transcript_id,
                        csv_file,
                        exc,
                    )
                    continue

                # Run feedback workflow
                try:
                    # Set the state in buddy
                    self.buddy.current_state = state

                    # Run the feedback workflow
                    result_state = self.buddy.end_session()

                    if result_state.get("error"):
                        logging.error(
                            "Evaluation failed for transcript %s in %s: %s",
                            transcript_id,
                            csv_file,
                            result_state["error"],
                        )
                        continue

                    # Extract results
                    # Get feedback from feedback_messages
                    feedback_messages = result_state.get("feedback_messages", [])
                    detailed_feedback = '\n'.join(
                        msg.content for msg in feedback_messages
                        if hasattr(msg, 'content')
                    )

                    results[str(transcript_id)] = {
                        "detailed_feedback": detailed_feedback,
                        "cod_stacer": result_state.get("cod_stacer", ""),
                        "saved_session_folder": result_state.get("folder_name", ""),
                        "source_file": str(csv_file),
                    }

                    logging.info(f"Successfully evaluated transcript {transcript_id}")

                except Exception as exc:
                    logging.exception(
                        "Error evaluating transcript %s in %s: %s",
                        transcript_id,
                        csv_file,
                        exc,
                    )
                    continue

        return results


def extract_case_feedback_data(root_path: str) -> pd.DataFrame:
    """
    Traverse all folders in `root_path`, extract Transcript ID, Quality Label,
    and Final Overall Score from the two text files (case_study.txt and MI_EVAL.txt),
    and return a DataFrame with columns:
        ['Transcript ID', 'Quality Label', 'Predicted Score'].
    """
    data = []

    for folder, _, files in os.walk(root_path):
        # Expect both files to be in the same folder
        case_study_path = os.path.join(folder, "case_study.txt")
        feedback_path = os.path.join(folder, "MI_EVAL.txt")

        transcript_id = None
        quality_label = None
        predicted_score = None

        # --- Extract from case_study.txt ---
        if os.path.exists(case_study_path):
            with open(case_study_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Extract "Transcript ID"
            match_id = re.search(r"Transcript ID:\s*(\d{0,3})", text)
            transcript_id = match_id.group(1) if match_id else None

            # Extract "Quality Label"
            match_quality = re.search(r"Quality Label:\s*(high|low)", text, re.IGNORECASE)
            quality_label = match_quality.group(1).lower() if match_quality else None

        # --- Extract from MI_EVAL.txt ---
        if os.path.exists(feedback_path):
            with open(feedback_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Try case 1: Overall MI Adherence: 3
            match_direct = re.search(r"Overall MI Adherence:\s*(\d+)", text)

            # Try range form followed by "Score:"
            match_score = re.search(r"Score:\s*(\d+)", text)

            # Try checkbox form (☑ 3 or ✓ 3)
            match_checkbox = re.search(r"[☑✓]\s*(\d+)", text)

            if match_checkbox:
                predicted_score = int(match_checkbox.group(1))
            elif match_score:
                predicted_score = int(match_score.group(1))
            elif match_direct:
                predicted_score = int(match_direct.group(1))
            else:
                predicted_score = None

        # Only add if at least one item was found
        if transcript_id or quality_label or predicted_score:
            data.append({
                "Transcript ID": transcript_id,
                "Quality Label": quality_label,
                "Predicted Score": predicted_score
            })

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Transcript ID", "Quality Label", "Predicted Score"])
    return df


def evaluate_accuracy(csv_path: str) -> float:
    """
    Evaluate prediction accuracy based on a CSV containing:
        - 'Quality Label' (values: 'high' or 'low')
        - 'Predicted Score' (integer values)

    Rules:
        - Correct if (Quality Label == 'high' and Predicted Score > 2)
        - Correct if (Quality Label == 'low' and Predicted Score <= 2)
        - Otherwise incorrect

    Returns:
        accuracy (float): Percentage of correct predictions
    """
    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"Quality Label", "Predicted Score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the columns: {required_cols}")

    # Calculate correctness per row
    df["Correct"] = (
            ((df["Quality Label"].str.lower() == "high") & (df["Predicted Score"] > 2))
            | ((df["Quality Label"].str.lower() == "low") & (df["Predicted Score"] <= 2))
    )

    # Tally results
    correct = df["Correct"].sum()
    total = len(df)
    incorrect = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Print summary
    print(f"✅ Total: {total}")
    print(f"✔️ Correct: {correct}")
    print(f"❌ Incorrect: {incorrect}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")

    return accuracy


def _prompt_for_value(env_key: str, prompt_text: str) -> str:
    """Prompt user for a value, checking environment first."""
    value = os.getenv(env_key, '').strip() if os.getenv(env_key) else ''
    if value:
        return value
    try:
        return input(prompt_text).strip()
    except EOFError:
        return ''


def _format_result(transcript_id: str, payload: Dict[str, str]) -> str:
    """Format evaluation result for display."""
    divider = '=' * 80
    parts = [
        divider,
        f"\nTranscript ID: {transcript_id} (source: {payload.get('source_file', 'unknown')})",
        '\n' + '-' * 80,
        '\nFilled CBT Form:',
        payload.get('cod_stacer', '(no CBT output)'),
        '\n',
        'Detailed Feedback:',
        payload.get('detailed_feedback', '(no feedback generated)'),
        '\n',
        f"Saved session folder: {payload.get('saved_session_folder', 'N/A')}\n"
    ]
    return ''.join(parts)


def main() -> None:
    """Main entry point for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get transcript directory
    # transcripts_dir = _prompt_for_value(
    #     'MI_TRANSCRIPTS_DIR',
    #     'Enter the path to the MI transcript CSV directory: '
    # )
    # if not transcripts_dir:
    #     print('No transcript directory provided; exiting.')
    #     return
    transcripts_dir = '/Users/rezabarzegar/PycharmProjects/LearningBuddy/Evaluation Testors/MI/Evaluator_test/Patient Data'

    # Get API key
    api_key = load_openai_api_key()

    # Initialize evaluator
    evaluator = TranscriptEvaluator(api_key=api_key, mode=MI_MODE)

    # Run evaluations
    try:
        results = evaluator.run_directory_evaluations(transcripts_dir)
    except Exception as exc:
        logging.exception('Evaluation run failed')
        print(f'Evaluation failed: {exc}')
        return

    if not results:
        print('No transcripts were processed. Verify the directory path and CSV schema.')
        return

    # Display results
    for transcript_id, payload in results.items():
        print(_format_result(transcript_id, payload))

    print('=' * 80)
    print('Evaluation complete.')

    # Extract metrics and calculate accuracy
    saved_sessions_path = "Saved_Sessions"
    if os.path.exists(saved_sessions_path):
        print("\nExtracting evaluation metrics...")
        df = extract_case_feedback_data(saved_sessions_path)

        metrics_csv = "Evaluation_Metric_Evaluator_Agent.csv"
        df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to: {metrics_csv}")

        if os.path.exists(metrics_csv):
            print("\nCalculating accuracy...")
            accuracy = evaluate_accuracy(metrics_csv)
            print(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()