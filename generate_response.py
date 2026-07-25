"""
Batch inference script for Music CRS.
"""

import argparse
from pathlib import Path

from environments import MusicCatalog, UserProfileDB
from datasets import load_dataset
from tqdm import tqdm
from llm_call.hf import HF_LLM_Client


BASE_DIR = Path(__file__).resolve().parent

JUDGE_SESSION_IDS = [
  "525f9f69-31e5-4fee-9813-e903ad24e39e" # randomly selected session ID for evaluation
]

def build_dialogue_context(conversations, music_db) -> str:
    """Build the model input from all turns preceding the target response."""
    dialogue_context = []
    for turn in conversations[:-1]:
        if turn["role"] == "music":
            role = "assistant"
            content = music_db.id_to_metadata(
                track_id=turn["content"],
            )
        else:
            role = turn["role"]
            content = turn["content"]
        dialogue_context.append({"role": role, "content": content})
    return dialogue_context


def main(args):
    prompt = """You are a music recommendation assistant. Generate an appropriate response about the track that has already been recommended, considering the recommended item, user demographics, and user query."""
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
    user_db = UserProfileDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        split_types=["all_users"],
    )
    llm_agents = HF_LLM_Client(model_name=args.llm_model, device=args.device)
    gt_dataset = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A-Master",
        split="test",
    )
    gt_dict = {item["session_id"]: item for item in gt_dataset}

    for sid in tqdm(sorted(JUDGE_SESSION_IDS), desc="Sessions"):
        gt_instance = gt_dict[sid]
        conversations = gt_instance["conversations"]
        user_id = gt_instance["user_id"]
        user_profile_str = user_db.id_to_profile_str(user_id)
        dialogue_context = build_dialogue_context(conversations, music_db)
        messages = [{"role": "system", "content": prompt + "\n\n" + user_profile_str}]
        messages.extend(dialogue_context)
        _, model_output_text = llm_agents.chat_completion(messages, max_new_tokens=1024)

        model_name = args.llm_model.replace("/", "-")
        output_path = BASE_DIR / "exp" / "response" / sid / f"{model_name}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(model_output_text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference on TalkPlayData-2 test dataset for Music CRS evaluation."
    )
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda).",
    )
    args = parser.parse_args()
    main(args)
