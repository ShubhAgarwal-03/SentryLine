"""Zero-shot environment classification via CLIP — PRD §9, Step 1.

No training required: CLIP already generalizes across scene types from its
pretraining. This is the mechanism that lets SentryLine be deployed to a
new site by writing a config file, not by retraining anything.
"""
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

CANDIDATE_ENVIRONMENTS = [
    "home",
    "hospital",
    "office",
    "factory floor",
    "retail store",
    "kitchen",
    "school",
]

_PROMPT_TEMPLATE = "a photo of a {}"


@lru_cache(maxsize=1)
def _load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


def classify_environment(image_rgb: np.ndarray) -> tuple[str, float]:
    """Returns (environment_label, confidence) — confidence is the softmax
    probability of the top match against CANDIDATE_ENVIRONMENTS.
    """
    model, preprocess, tokenizer = _load_model()

    pil_image = Image.fromarray(image_rgb)
    image_input = preprocess(pil_image).unsqueeze(0)
    text_inputs = tokenizer([_PROMPT_TEMPLATE.format(env) for env in CANDIDATE_ENVIRONMENTS])

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_prob, top_idx = similarity[0].max(dim=0)

    return CANDIDATE_ENVIRONMENTS[int(top_idx)], float(top_prob)