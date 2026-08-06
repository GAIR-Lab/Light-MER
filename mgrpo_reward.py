"""Semantic and format rewards used by Light-MER Stage 2 M-GRPO training."""

import re
from typing import Dict, List, Set, Tuple

TARGET_PREFIX = "The character's emotional state is "

_CANONICAL = {
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "pleased": "happy",
    "sadness": "sad",
    "unhappy": "sad",
    "anger": "angry",
    "mad": "angry",
    "fear": "afraid",
    "fearful": "afraid",
    "scared": "afraid",
    "worry": "worried",
    "worrying": "worried",
    "anxiety": "anxious",
    "surprised": "surprise",
    "disgusted": "disgust",
    "neutrality": "neutral",
}

_STOPWORDS = {
    "the", "character", "person", "speaker", "emotional", "emotion", "state",
    "states", "is", "are", "feels", "feel", "feeling", "appears", "seems",
    "shows", "showing", "and", "or", "with", "of", "a", "an", "to", "be",
    "likely", "possibly", "might", "may", "main", "identifiable", "clearly",
}

_REASONING_PATTERNS = [
    r"\bbased on\b", r"\bwe can infer\b", r"\bsuggests? that\b",
    r"\bindicates?\b", r"\btherefore\b", r"\bin the text\b",
    r"\bthe caption reads\b", r"\baudio cues?\b", r"\bvideo clues?\b",
    r"\bfacial expression\b", r"\btone of voice\b", r"\bfrom the\b",
    r"\bconsidering\b", r"\bthis sentence\b", r"\bthe speaker\b",
    r"\bwe can see\b", r"\bit can be\b", r"\bbecause\b",
]


def _canonicalize(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"[^a-z\-\s]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    if not label:
        return ""
    words = [w for w in label.split() if w not in _STOPWORDS]
    if not words:
        return ""
    if len(words) > 3:
        words = words[:3]
    label = " ".join(words)
    return _CANONICAL.get(label, label)


def _split_items(text: str) -> List[str]:
    text = text.replace("[", " ").replace("]", " ")
    text = re.sub(r"\b(and|or)\b", ",", text, flags=re.IGNORECASE)
    return [x.strip() for x in re.split(r"[,;/]", text) if x.strip()]


def extract_pred_emotions(raw_response: str) -> Set[str]:
    raw = raw_response.strip()
    raw = raw.split("Assistant:")[-1].strip().replace("###", " ").strip()

    if raw.startswith(TARGET_PREFIX):
        candidate = raw[len(TARGET_PREFIX):]
    else:
        m = re.search(
            r"(?:emotional states?\s+(?:is|are)|emotion(?:s)?\s+(?:is|are)|feels?|feeling|appears|seems)[:\s]+(.+)",
            raw,
            flags=re.IGNORECASE,
        )
        candidate = m.group(1) if m else raw

    candidate = re.split(
        r"(?:\.|\n|\bbecause\b|\bbased on\b|\bdue to\b|\bas shown\b|\bas indicated\b)",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    preds = {_canonicalize(item) for item in _split_items(candidate)}
    return {p for p in preds if p}


def extract_gt_emotions(gt_emotions: str) -> Set[str]:
    return {p for p in (_canonicalize(item) for item in _split_items(str(gt_emotions))) if p}


def reward_format(raw_response: str) -> float:
    raw = raw_response.strip()
    if not raw.startswith(TARGET_PREFIX):
        return 0.0
    emotion_part = raw[len(TARGET_PREFIX):]
    first_period = emotion_part.find(".")
    if first_period == -1:
        return 0.2
    after_period = emotion_part[first_period + 1:].strip()
    emotions = extract_pred_emotions(raw)
    if not emotions:
        return 0.2
    if len(after_period) == 0:
        return 1.0
    if len(after_period) <= 5:
        return 0.5
    return 0.0


def reward_brevity(raw_response: str, n_generated_tokens: int = 0) -> float:
    word_count = len(raw_response.strip().split())
    if word_count <= 18:
        return 1.0
    if word_count <= 35:
        return 0.7
    if word_count <= 60:
        return 0.3
    return 0.0


def reward_no_reasoning(raw_response: str) -> float:
    raw_lower = raw_response.lower()
    for pattern in _REASONING_PATTERNS:
        if re.search(pattern, raw_lower):
            return 0.0
    return 1.0


def reward_clean_stop(raw_response: str) -> float:
    raw = raw_response.strip()
    sentences = [s.strip() for s in raw.split(".") if s.strip()]
    if len(sentences) == 1:
        return 1.0
    if len(sentences) == 2:
        return 0.4
    return 0.0


def reward_emotion_overlap(raw_response: str, gt_emotions: str) -> float:
    pred_emotions = extract_pred_emotions(raw_response)
    gt_set = extract_gt_emotions(gt_emotions)
    if not pred_emotions or not gt_set:
        return 0.0
    intersection = pred_emotions & gt_set
    precision = len(intersection) / len(pred_emotions)
    recall = len(intersection) / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_label_count(raw_response: str) -> float:
    n = len(extract_pred_emotions(raw_response))
    if 1 <= n <= 5:
        return 1.0
    if 6 <= n <= 8:
        return 0.5
    return 0.0


def compute_reward_with_components(
    raw_response: str,
    gt_emotions: str = None,
    n_generated_tokens: int = 0,
    w_format: float = 1.0,
    w_brevity: float = 0.5,
    w_no_reason: float = 0.75,
    w_clean_stop: float = 0.5,
    w_emotion: float = 6.0,
) -> Tuple[float, Dict[str, float]]:
    r_format = reward_format(raw_response)
    r_brevity = reward_brevity(raw_response, n_generated_tokens)
    r_no_reason = reward_no_reasoning(raw_response)
    r_clean_stop = reward_clean_stop(raw_response)
    r_count = reward_label_count(raw_response)
    r_emotion = reward_emotion_overlap(raw_response, gt_emotions) if gt_emotions is not None else 0.0

    total = (
        w_format * r_format
        + w_brevity * r_brevity
        + w_no_reason * r_no_reason
        + w_clean_stop * r_clean_stop
        + w_emotion * r_emotion
        + 0.25 * w_emotion * r_count
    )
    components = {
        "format": r_format,
        "brevity": r_brevity,
        "no_reason": r_no_reason,
        "clean_stop": r_clean_stop,
        "emotion_f1": r_emotion,
        "label_count": r_count,
        "total": total,
    }
    return total, components


def compute_reward(*args, **kwargs) -> float:
    total, _ = compute_reward_with_components(*args, **kwargs)
    return total
