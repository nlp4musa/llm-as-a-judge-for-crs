import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from environments import MusicCatalog
from generate_response import JUDGE_SESSION_IDS, build_dialogue_context
from llm_call.gemini import Gemini_LLM_Client

BASE_DIR = Path(__file__).resolve().parent
TARGET_MODELS = (
    "google-gemma-4-E2B-it",
    "google-gemma-4-E4B-it",
    "meta-llama-Llama-3.2-1B-Instruct",
    "meta-llama-Llama-3.2-3B-Instruct",
)
REQUIRED_SCORE_FIELDS = (
    "personalization_score",
    "personalization_reason",
    "explanation_score",
    "explanation_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full-context LLM-as-a-Judge evaluation."
    )
    parser.add_argument(
        "--judge_model",
        default="gemini-3.1-flash-lite-preview",
        help="Gemini judge model identifier.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum API/JSON attempts for each response (default: 3).",
    )
    args = parser.parse_args()
    if args.max_retries < 1:
        parser.error("--max_retries must be at least 1")
    return args


def llm_as_a_judge(
    client: Any,
    judge_prompt_template: str,
    user_profile: str,
    dialogue_context: str,
    assistant_response: str,
    max_retries: int,
) -> dict:
    """Call an LLM judge and return a validated score dictionary."""
    prompt = (
        judge_prompt_template
        .replace("{user_profile}", user_profile)
        .replace("{dialogue_context}", dialogue_context)
        .replace("{assistant_response}", assistant_response)
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            score = json.loads(client.chat_completion(prompt))
            missing = [key for key in REQUIRED_SCORE_FIELDS if key not in score]
            if missing:
                raise ValueError(f"Missing score fields: {', '.join(missing)}")
            for key in ("personalization_score", "explanation_score"):
                if type(score[key]) is not int or not 1 <= score[key] <= 5:
                    raise ValueError(f"{key} must be an integer from 1 to 5")
            return score
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Judge failed after {max_retries} attempts: {last_error}"
    ) from last_error


def build_client(judge_model: str) -> Gemini_LLM_Client:
    load_dotenv(BASE_DIR / "dotenv")
    return Gemini_LLM_Client(
        model_name=judge_model,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


def main() -> None:
    args = parse_args()
    client = build_client(args.judge_model)
    prompt_path = BASE_DIR / "prompt" / "judge_prompt.txt"
    with prompt_path.open(encoding="utf-8") as prompt_file:
        judge_prompt_template = prompt_file.read()

    music_db = MusicCatalog(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=[
            "track_name",
            "artist_name",
            "album_name",
            "release_date",
            "tag_list",
        ],
    )
    source_dataset = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A-Master",
        split="test",
    )
    source_by_session = {item["session_id"]: item for item in source_dataset}
    judge_model_tag = args.judge_model.replace("/", "-")

    for session_id in tqdm(sorted(JUDGE_SESSION_IDS), desc="Sessions"):
        item = source_by_session[session_id]
        user_profile = item["user_profile"]
        dialogue_context = build_dialogue_context(item["conversations"], music_db)
        for target_model in TARGET_MODELS:
            response_path = (
                BASE_DIR / "exp" / "response" / session_id / f"{target_model}.txt"
            )
            if not response_path.is_file():
                raise FileNotFoundError(
                    f"Missing generated response: {response_path}. "
                    "Run generate_response.py for all target models first."
                )
            assistant_response = response_path.read_text(encoding="utf-8").strip()

            output_path = (
                BASE_DIR
                / "exp"
                / "judge_score"
                / judge_model_tag
                / session_id
                / f"{target_model}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                continue

            score = llm_as_a_judge(
                client=client,
                judge_prompt_template=judge_prompt_template,
                user_profile=str(user_profile),
                dialogue_context=str(dialogue_context),
                assistant_response=str(assistant_response),
                max_retries=args.max_retries,
            )
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(score, output_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
