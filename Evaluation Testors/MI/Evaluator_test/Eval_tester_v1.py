import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import pandas as pd
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Import the new modular LearningBuddy
from LearningBuddy_Agent import LearningBuddy, MI_MODE, STACER_MODE
from LearningBuddy_Agent.graph_state import LBState, PatientContext, SessionMetadata
from utils.api_key_loader import load_openai_api_key

# --- New Imports for Full Evaluation ---
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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
                print('========================')
                print(f"Processing transcript file: {csv_file}")
                print('========================')
            except Exception as exc:
                logging.exception("Failed to read transcript file %s: %s", csv_file, exc)
                continue

            if df.empty:
                logging.error("Skipping empty transcript file %s", csv_file)
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


import pandas as pd
import os
import re


def extract_case_feedback_data(root_path: str) -> pd.DataFrame:
    """
    Traverse all folders in `root_path`, extract Transcript ID, Quality Label,
    and Final Overall Score from the two text files (case_study.txt and MI_EVAL.txt),
    and return a DataFrame with columns:
        ['Transcript ID', 'Quality Label', 'Predicted Score'].

    The function uses robust regular expressions to handle varied scoring formats
    in MI_EVAL.txt, including decimals (which are rounded to the nearest integer).
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

            predicted_score = None

            # 1. Look for a score expressed as a fraction (e.g., '3/5', '1/5')
            # This captures the numerator (the desired score).
            # Captures '3' from 'Score: 3/5' or 'Comments: 3/5'
            match_fraction = re.search(r"(\d)/5", text)

            # 2. Look for explicit score labels or checkboxes (e.g., '[X] 3', 'Score: 3')
            # The (?:\b...|\b...) non-capturing group looks for a leading keyword or symbol
            # The \d captures the single-digit score
            match_score_label = re.search(r"(?:[☑✓]|\bScore|\bRating)[:\s]*(\d)", text, re.IGNORECASE)

            # 3. Look for the score right after the main header 'Overall MI Adherence:'
            # Captures the first digit/decimal number (e.g., '3', '3.56') near the header.
            # re.DOTALL allows the .* to match across newlines
            match_header = re.search(r"Overall MI Adherence:.*?(\d+\.?\d*)", text, re.DOTALL | re.IGNORECASE)

            # --- Prioritized Assignment ---
            score_str = None

            if match_fraction:
                # Highest priority for clean X/5 format
                score_str = match_fraction.group(1)
            elif match_score_label:
                # Second priority for explicit label/checkbox score
                score_str = match_score_label.group(1)
            elif match_header:
                # Third priority for decimal or number next to the main title
                score_str = match_header.group(1)

            if score_str:
                try:
                    score_float = float(score_str)

                    # Round the float to the nearest integer score (e.g., 3.56 -> 4, 1.11 -> 1)
                    # This handles both whole numbers (3 -> 3) and decimals.
                    predicted_score = int(round(score_float))
                except ValueError:
                    # Should not happen if regex works, but safe to catch
                    predicted_score = None
            else:
                predicted_score = None

        # Only add if at least one piece of useful data was found
        if transcript_id or quality_label or predicted_score:
            data.append({
                "Transcript ID": transcript_id,
                "Quality Label": quality_label,
                "Predicted Score": predicted_score
            })

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Transcript ID", "Quality Label", "Predicted Score"])
    return df

def run_full_evaluation_analysis(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Evaluates prediction accuracy by varying the threshold from 1 to 4.
    Computes confusion matrix, accuracy, precision, recall, and F1-score
    for each threshold.

    Args:
        csv_path: Path to the CSV file containing 'Quality Label' and 'Predicted Score'.

    Returns:
        A DataFrame containing the summary of metrics for each threshold, or None.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Metrics file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

    required_cols = {"Quality Label", "Predicted Score"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain the columns: {required_cols}")
        return None

    # Drop rows with missing labels or scores, as they can't be evaluated
    df = df.dropna(subset=['Quality Label', 'Predicted Score'])
    df['Predicted Score'] = pd.to_numeric(df['Predicted Score'], errors='coerce')
    df = df.dropna(subset=['Predicted Score'])
    df['Predicted Score'] = df['Predicted Score'].astype(int)

    if df.empty:
        print("No valid data to evaluate after dropping missing values.")
        return None

    y_true = df['Quality Label'].str.lower()
    y_score = df['Predicted Score']

    print("\n" + "=" * 80)
    print("Running Full Evaluation Analysis (Positive Label: 'high')")
    print("=" * 80 + "\n")

    results_data = []

    # Define the order of labels for confusion matrix and metrics
    # 'high' is the "positive" class, 'low' is the "negative" class
    metric_labels = ['high', 'low']
    pos_label = 'high'

    # Threshold T means scores [1...T] are 'low', [T+1...5] are 'high'
    for threshold in range(1, 5):  # This loops T=1, T=2, T=3, T=4

        # Classify scores: > threshold is 'high', <= threshold is 'low'
        y_pred = y_score.apply(lambda x: 'high' if x > threshold else 'low')

        # --- Calculate Metrics ---
        # We set zero_division=0 to handle cases where a denominator is zero
        # (e.g., no 'high' predictions -> precision=0)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=metric_labels, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=metric_labels, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=metric_labels, pos_label=pos_label, zero_division=0)

        # Get confusion matrix
        # labels=['high', 'low'] means:
        #          Pred: high | Pred: low
        # True: high   [[  TP   ,    FN   ],
        # True: low    [   FP   ,    TN   ]]
        cm = confusion_matrix(y_true, y_pred, labels=metric_labels)

        # Handle cases where matrix isn't 2x2 (e.g., only one class present/predicted)
        if cm.shape == (2, 2):
            tp, fn, fp, tn = cm.ravel()
        else:
            # Re-calculate manually for robustness if cm is not 2x2
            tp = ((y_pred == 'high') & (y_true == 'high')).sum()
            fn = ((y_pred == 'low') & (y_true == 'high')).sum()
            fp = ((y_pred == 'high') & (y_true == 'low')).sum()
            tn = ((y_pred == 'low') & (y_true == 'low')).sum()

        # --- Print Report for this Threshold ---
        print(f"--- Threshold: {threshold} (Scores {threshold + 1}-5 = 'high' | Scores 1-{threshold} = 'low') ---")
        print(f"  Confusion Matrix (Rows: True, Cols: Pred; Labels: {metric_labels})")
        print(f"    [[TP: {tp:^4}, FN: {fn:^4}]")
        print(f"     [FP: {fp:^4}, TN: {tn:^4}]]\n")
        print(f"  Accuracy:   {accuracy: .4f}")
        print(f"  Precision:  {precision:.4f}  (TP / (TP + FP))")
        print(f"  Recall:     {recall:.4f}  (TP / (TP + FN))")
        print(f"  F1-Score:   {f1:.4f}\n")

        results_data.append({
            "Threshold": threshold,
            "Rule (High)": f">{threshold}",
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    # --- Print Summary Table ---
    results_df = pd.DataFrame(results_data)
    print("=" * 80)
    print("Summary of Metrics Across All Thresholds")
    print("=" * 80)

    # Set display options for cleaner table printing
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 1000,
            'display.float_format', '{:,.4f}'.format
    ):
        print(results_df.to_string(index=False))

    return results_df


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

    # # Initialize evaluator
    # evaluator = TranscriptEvaluator(api_key=api_key, mode=MI_MODE)
    #
    # # Run evaluations
    # try:
    #     results = evaluator.run_directory_evaluations(transcripts_dir)
    # except Exception as exc:
    #     logging.exception('Evaluation run failed')
    #     print(f'Evaluation failed: {exc}')
    #     return
    #
    # if not results:
    #     print('No transcripts were processed. Verify the directory path and CSV schema.')
    #     return
    #
    # # Display results
    # for transcript_id, payload in results.items():
    #     print(_format_result(transcript_id, payload))
    #
    # print('=' * 80)
    # print('Evaluation complete.')


    # Extract metrics and calculate accuracy
    saved_sessions_path = "Saved_Sessions"
    if os.path.exists(saved_sessions_path):
        print("\nExtracting evaluation metrics...")
        df = extract_case_feedback_data(saved_sessions_path)

        metrics_csv = "Evaluation_Metric_Evaluator_Agent.csv"
        df.to_csv(metrics_csv, index=False)
        print(f"Metrics saved to: {metrics_csv}")

        if os.path.exists(metrics_csv):
            print("\nCalculating accuracy across all thresholds...")
            # Run the new, full analysis
            run_full_evaluation_analysis(metrics_csv)


if __name__ == '__main__':
    main()