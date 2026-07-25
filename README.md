# LLM-as-a-Judge for Conversational Music Recommendation

**LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation**

> Seungheon Doh, Bruno Sguerra, Sergio Oramas, Elena V. Epure, and Juhan Nam.

> Accepted as a short paper at the **20th ACM Conference on Recommender Systems (RecSys '26)**, held September 28–October 2, 2026, in Minneapolis, Minnesota, USA.


## Overview

Conversational recommender systems must not only retrieve relevant items but also generate responses that explain and personalize their recommendations. Traditional reference-based text metrics are poorly suited to this setting because many valid responses may differ substantially from a single reference and because response quality depends on the full interaction context.

This work studies whether an LLM can serve as a reliable evaluator of conversational music recommendation responses. Our framework evaluates two independent dimensions:

- **Personalization Quality:** how well a response reflects the user's request, conversation history, and profile.
- **Explanation Quality:** how clearly the response connects the recommended item and its musical attributes to the user's needs.

The study covers 20 multi-turn recommendation sessions and responses from four open-source instruction-tuned models. We collect 400 ratings from 20 music-domain experts and compare human judgments with reference-based metrics and contextual LLM judges. The results show that LLM judges align more closely with human assessments than traditional reference-based metrics, while the remaining moderate correlation indicates that human oversight is still important.

The released pipeline has two stages:

1. Generate conversational recommendation responses with open-source models.
2. Evaluate each response using an LLM judge conditioned on the user profile, full dialogue context, recommended-track metadata, and the original domain-specific rubric.

## Links

- **arXiv:** Coming soon — the link will be added after the preprint is released.
- **Conference:** [20th ACM Conference on Recommender Systems (RecSys '26)](https://recsys.acm.org/recsys26/)
- **Proceedings/DOI:** Coming soon.

## Setup

Python 3.10 or newer and a CUDA GPU are recommended.

```bash
cd open_source
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The scripts load these TalkPlay datasets from Hugging Face:

- `talkpl-ai/TalkPlayData-Challenge-Blind-A-Master`
- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`
- `talkpl-ai/TalkPlayData-Challenge-User-Metadata`

Make sure they are accessible before running the pipeline.

## 1. Generate responses with open-source models

Run all four response models:

```bash
bash scripts/infer.sh
```

Or run one model:

```bash
python generate_response.py \
  --llm_model meta-llama/Llama-3.2-1B-Instruct
```

Use `--device cpu` when CUDA is unavailable:

```bash
python generate_response.py \
  --llm_model meta-llama/Llama-3.2-1B-Instruct \
  --device cpu
```

Responses are saved to:

```text
exp/response/{session_id}/{model_name}.txt
```

### Verified inference example

We successfully ran `meta-llama/Llama-3.2-1B-Instruct` on the included example session (`525f9f69-31e5-4fee-9813-e903ad24e39e`) using CPU inference. The generated response was saved to:

```text
exp/response/525f9f69-31e5-4fee-9813-e903ad24e39e/meta-llama-Llama-3.2-1B-Instruct.txt
```

<details>
<summary>Example generated response</summary>

> I'm glad I could help you identify the album you were thinking of. If you're looking for more Finnish melodic death metal recommendations, I've got some great ones. Here are a few more:
> **Blind Guardian** — Their early work, such as "Twilight of the Thunder God" or "Hearts and Bones", has a similar epic and melodic vibe.
> Let me know if you'd like more recommendations or if you have any specific preferences (e.g., more aggressive, more melodic, etc.)!

</details>

Generation uses sampling, so the exact response may differ across runs.

## 2. Run the conversational LLM-as-a-Judge

Set the Gemini API key:

```bash
export GEMINI_API_KEY="..."
```

Run the final full-context judge:

```bash
python eval_llm_as_a_judge.py \
  --judge_model gemini-3.1-flash-lite-preview
```

For every response, the judge receives:

- the user profile;
- the complete conversation before the target response;
- recommended-track metadata contained in that conversation; and
- the generated assistant response.

The original full prompt is stored at `prompt/judge_prompt.txt`. Outputs are saved without score post-processing:

```text
exp/judge_score/{judge_model}/{session_id}/{response_model}.json
```

Existing output files are skipped, so an interrupted run can be resumed.

## Output schema

```json
{
  "personalization_score": int,
  "personalization_reason": str,
  "explanation_score": int,
  "explanation_reason": str
}
```

API keys, datasets, generated responses, and judge outputs are not included in the repository.


## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{doh2026llmjudge,
  author    = {Doh, Seungheon and Sguerra, Bruno and Oramas, Sergio and Epure, Elena V. and Nam, Juhan},
  title     = {{LLM-as-a-Judge} for Evaluating System Responses in Conversational Music Recommendation},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  series    = {RecSys '26},
  year      = {2026},
  location  = {Minneapolis, MN, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  note      = {Accepted short paper. DOI and page numbers forthcoming}
}
```
