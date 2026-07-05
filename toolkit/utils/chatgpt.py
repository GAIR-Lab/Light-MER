"""No-network stubs kept only for compatibility with legacy imports.

The open-source stage1 package does not use OpenAI or other hosted GPT APIs.
Use local Hugging Face/vLLM models for evaluation label extraction.
"""


def _removed_api(*args, **kwargs):
    raise RuntimeError(
        "Hosted GPT API helpers were removed from the SWD-H stage1 open-source package."
    )


func_get_completion = _removed_api
get_completion = _removed_api
get_translate_eng2chi = _removed_api
get_translate_chi2eng = _removed_api
get_image_emotion_batch = _removed_api
get_video_emotion_batch = _removed_api
get_text_emotion_batch = _removed_api
get_multi_emotion_batch = _removed_api
get_evoke_emotion_batch = _removed_api
get_micro_emotion_batch = _removed_api
