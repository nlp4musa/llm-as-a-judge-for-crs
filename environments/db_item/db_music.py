import os
import torch
import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from collections import Counter
import random

class MusicCatalog:
    def __init__(self,
            dataset_name: str = "talkpl-ai/TalkPlayData-2-Track-Metadata",
            split_types: list[str] = ["test_warm", "test_cold"],
            corpus_types: list[str] = ["track_name", "artist_name", "album_name", "release_date", "tag_list"],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        all_tag_list_of_list = metadata_concat_dataset['tag_list']
        flatten_tag_list = [tag for sublist in tqdm(all_tag_list_of_list) for tag in sublist]
        tag_count = Counter(flatten_tag_list)
        self.important_tags = [tag for tag, count in tag_count.items() if count > 50]
        print(f"There are {len(self.important_tags)} important tags")

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        metadata = self.metadata_dict[track_id]
        track_id = metadata['track_id']
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            values = metadata[corpus_type]
            if corpus_type == "tag_list":
                filted_value = [tag for tag in set(values) if tag in self.important_tags]
                values = random.sample(filted_value, min(len(filted_value), 10))
            if isinstance(values, list) and len(values) > 0:
                corpus_type_value = ", ".join(values)
            elif isinstance(values, str):
                corpus_type_value = values
            else:
                corpus_type_value = str(values)
            entity_str += f", {corpus_type}: {corpus_type_value}"
        return entity_str
