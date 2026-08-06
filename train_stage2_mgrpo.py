"""Stage 2 M-GRPO training for the Light-MER Qwen3-0.6B student."""


import ast
import os
import copy
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.datasets.builders.image_text_pair_builder import *
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.processors.video_processor import load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
from mgrpo_reward import compute_reward_with_components, reward_format

import config as app_config


def build_prompt(face_or_frame, subtitle, user_message):
    """Standalone prompt builder mirroring BaseDataset.get_prompt_for_multimodal."""
    if face_or_frame == 'multiface_audio_face_text':
        prompt = (f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. "
                  f"The audio content is as follows: <Audio><AudioHere></Audio>. "
                  f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "
                  f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. "
                  f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: ")
    elif face_or_frame == 'face':
        prompt = (f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. "
                  f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "
                  f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. "
                  f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: ")
    elif face_or_frame == 'multiface_text':
        prompt = (f"###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. "
                  f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. "
                  f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: ")
    else:
        prompt = (f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. "
                  f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. "
                  f"Now, please answer my question based on all the provided information. {user_message} ###Assistant: ")
    return prompt


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_emotion_labels(value):
    """Parse MER-Caption+ label lists without importing legacy toolkit code."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value == "" or pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [
        item.strip().strip("'\"")
        for item in text.strip("[]").split(",")
        if item.strip().strip("'\"")
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Light-MER Stage 2 M-GRPO training")
    parser.add_argument("--cfg-path", required=True, help="Path to the Stage 2 config YAML")
    parser.add_argument("--options", nargs="+", help="Override config params")
    args = parser.parse_args()
    return args


def load_training_annotations():
    """Load MERCaptionPlus training data annotations (name, subtitle, ovlabel)."""
    ov_path = os.path.join(app_config.DATA_DIR['MERCaptionPlus'], 'track2_train_mercaptionplus.csv')
    if not os.path.isfile(ov_path):
        raise FileNotFoundError(f"Missing MER-Caption+ labels: {ov_path}")

    name2openset = {}
    df = pd.read_csv(ov_path)
    for _, row in df.iterrows():
        name = row['name']
        openset = row['openset']
        openset = parse_emotion_labels(openset)
        if len(openset) == 0:
            openset = ['neutral']
        name2openset[name] = ", ".join(openset)

    subtitle_csv = app_config.PATH_TO_TRANSCRIPTIONS['MERCaptionPlus']
    if not os.path.isfile(subtitle_csv):
        raise FileNotFoundError(f"Missing MER-Caption+ subtitles: {subtitle_csv}")

    name2subtitle = {}
    df = pd.read_csv(subtitle_csv)
    for _, row in df.iterrows():
        name = row['name']
        subtitle = row['english']
        if pd.isna(subtitle):
            subtitle = ""
        name2subtitle[name] = subtitle

    wav_root = app_config.PATH_TO_RAW_AUDIO['MERCaptionPlus']
    face_root = app_config.PATH_TO_RAW_FACE['MERCaptionPlus']

    annotations = []
    for name in name2openset:
        annotations.append({
            'name': name,
            'subtitle': name2subtitle.get(name, ''),
            'ovlabel': name2openset[name],
        })

    return annotations, wav_root, face_root


def load_sample_data(sample, wav_root, face_root, model_cfg, vis_processor):
    """Load raw multimodal data for one sample (audio, face)."""
    name = sample['name']
    n_frms = model_cfg.vis_processor.train.n_frms

    audio_path = os.path.join(wav_root, name + '.wav')
    face_path = os.path.join(face_root, name, name + '.npy')

    sample_data = {
        'face': None, 'raw_face': None,
        'frame': None, 'raw_frame': None,
        'audio': None, 'raw_audio': None,
        'image': None, 'raw_image': None,
    }

    if os.path.exists(face_path):
        raw_face, _ = load_face(face_path, n_frms=n_frms, height=224, width=224,
                                sampling="uniform", return_msg=True)
        face = vis_processor.transform(raw_face)
        sample_data['face'] = face
        sample_data['raw_face'] = raw_face

    if os.path.exists(audio_path):
        raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
        audio = transform_audio(raw_audio, "cpu")
        sample_data['audio'] = audio
        sample_data['raw_audio'] = raw_audio

    return sample_data


@torch.no_grad()
def generate_completions(chat, prompt, img_list, num_completions=4,
                         max_new_tokens=300, temperature=1.0, top_p=0.9):
    """
    Generate K completions WITHOUT ### stopping criteria.

    Key insight: During GRPO, we MUST let the model generate freely so we can
    observe whether it produces reasoning text after the emotion sentence.
    If we stop at ###, the model always looks "perfect" and GRPO has no signal.

    The stopping criteria is temporarily replaced with EOS-only, then restored.
    """
    from transformers import StoppingCriteriaList
    from my_affectgpt.conversation.conversation_video import StoppingCriteriaSub

    saved_criteria = chat.stopping_criteria
    eos_only = StoppingCriteriaList([StoppingCriteriaSub(
        stops=[torch.tensor([chat.tokenizer.eos_token_id]).to(chat.device)]
    )])
    chat.stopping_criteria = eos_only

    completions = []       # processed (after ### truncation)
    full_decoded_list = [] # completely unprocessed decoded text
    token_counts = []
    chat._return_raw = True
    try:
        for _ in range(num_completions):
            processed, full_decoded, n_tokens = chat.answer_sample(
                prompt=prompt, img_list=img_list,
                num_beams=1, temperature=temperature,
                do_sample=True, top_p=top_p,
                max_new_tokens=max_new_tokens, max_length=2000,
                print_prompt=False)
            completions.append(processed)
            full_decoded_list.append(full_decoded)
            token_counts.append(n_tokens)
    finally:
        chat._return_raw = False
        chat.stopping_criteria = saved_criteria
    return completions, full_decoded_list, token_counts


def llm_embed_tokens(model, input_ids):
    """Resolve token embeddings for the PEFT-wrapped Qwen model."""
    for attrs in [
        ("model", "model", "embed_tokens"),
        ("model", "embed_tokens"),
        ("base_model", "model", "model", "embed_tokens"),
    ]:
        module = model.llama_model
        try:
            for attr in attrs:
                module = getattr(module, attr)
        except AttributeError:
            continue
        return module(input_ids)
    raise AttributeError("Could not find the LLM token embedding module")


def build_input_embeds(model, chat, prompt, img_list, device):
    """
    Build input embeddings with multimodal tokens replaced,
    mirroring the Chat.answer_sample logic but returning embeds + mask.
    """
    IMAGE_PATCH_TOKEN_ID = chat.tokenizer.get_vocab()[app_config.DEFAULT_IMAGE_PATCH_TOKEN]
    AUDIO_PATCH_TOKEN_ID = chat.tokenizer.get_vocab()[app_config.DEFAULT_AUDIO_PATCH_TOKEN]
    FRAME_PATCH_TOKEN_ID = chat.tokenizer.get_vocab()[app_config.DEFAULT_FRAME_PATCH_TOKEN]
    FACE_PATCH_TOKEN_ID = chat.tokenizer.get_vocab()[app_config.DEFAULT_FACE_PATCH_TOKEN]
    MULTI_PATCH_TOKEN_ID = chat.tokenizer.get_vocab()[app_config.DEFAULT_MULTI_PATCH_TOKEN]

    replaced_prompt = chat.replace_token_for_multimodal(prompt)
    input_id = chat.to_token_ids(replaced_prompt, max_length=2000)

    attention_mask = input_id.ne(chat.tokenizer.pad_token_id).to(device)

    temp_input_id = copy.deepcopy(input_id).to(device)
    temp_input_id[temp_input_id == FRAME_PATCH_TOKEN_ID] = 0
    temp_input_id[temp_input_id == FACE_PATCH_TOKEN_ID] = 0
    temp_input_id[temp_input_id == AUDIO_PATCH_TOKEN_ID] = 0
    temp_input_id[temp_input_id == MULTI_PATCH_TOKEN_ID] = 0
    temp_input_id[temp_input_id == IMAGE_PATCH_TOKEN_ID] = 0
    cur_input_embeds = llm_embed_tokens(model, temp_input_id)
    cur_input_ids = input_id.to(device)

    cur_idx = 0
    for (patch_token_id, query_token_number, embeds) in [
        (FRAME_PATCH_TOKEN_ID, chat.num_video_query_token, img_list.get('frame')),
        (FACE_PATCH_TOKEN_ID, chat.num_video_query_token, img_list.get('face')),
        (AUDIO_PATCH_TOKEN_ID, chat.num_audio_query_token, img_list.get('audio')),
        (MULTI_PATCH_TOKEN_ID, chat.num_multi_query_token, img_list.get('multi')),
        (IMAGE_PATCH_TOKEN_ID, chat.num_image_query_token, img_list.get('image')),
    ]:
        if (cur_input_ids == patch_token_id).sum() != 0:
            if embeds is None:
                continue
            cur_features = embeds[cur_idx]
            masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
            mask_index_start = masked_indices[0]
            cur_input_embeds = torch.cat((
                cur_input_embeds[:mask_index_start],
                cur_features,
                cur_input_embeds[mask_index_start + query_token_number:]), dim=0)

    input_embeds = cur_input_embeds.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    return input_embeds, attention_mask


def compute_sequence_log_probs(model, input_embeds, attention_mask,
                               response_ids, autocast_dtype):
    """
    Compute per-token log probabilities of response tokens given the context.
    Returns: sum of log probs over all response tokens (scalar).
    """
    device = input_embeds.device
    response_embeds = llm_embed_tokens(model, response_ids)
    full_embeds = torch.cat([input_embeds, response_embeds], dim=1)

    input_len = input_embeds.shape[1]
    resp_len = response_ids.shape[1]
    full_attn = torch.ones(1, input_len + resp_len, device=device, dtype=attention_mask.dtype)
    full_attn[:, :input_len] = attention_mask

    with autocast(dtype=autocast_dtype):
        outputs = model.llama_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            return_dict=True)

    logits = outputs.logits[:, input_len - 1: input_len + resp_len - 1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1).squeeze(0)


def grpo_loss(policy_log_probs, ref_log_probs, advantages, clip_eps=0.2, kl_coef=0.01):
    """
    Stable GRPO clipped surrogate loss.

    The older script used exp(policy-ref) directly on sequence log-prob sums,
    which can create very large ratios. This version clamps log-ratios and uses
    the standard non-negative approximate KL: (ratio - 1) - log_ratio.
    """
    log_ratio = policy_log_probs - ref_log_probs
    log_ratio_clamped = torch.clamp(log_ratio, -5.0, 5.0)
    ratio = torch.exp(log_ratio_clamped)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    approx_kl = (ratio - 1.0) - log_ratio_clamped
    loss = -surrogate.mean() + kl_coef * approx_kl.mean()
    stats = {
        'log_ratio_mean': log_ratio.detach().mean().item(),
        'log_ratio_absmax': log_ratio.detach().abs().max().item(),
        'approx_kl': approx_kl.detach().mean().item(),
    }
    return loss, stats


def main():
    args = parse_args()
    cfg = Config(args)

    grpo_cfg = cfg.model_cfg.get("grpo", {})
    seed = cfg.run_cfg.get("seed", 42)
    setup_seeds(seed)
    setup_logger()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 M-GRPO training requires a CUDA GPU")
    device = torch.device("cuda:0")

    # ====== Build model from the Stage 1 SWD-H checkpoint ======
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    ckpt_path = grpo_cfg.get("ckpt", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading Stage 1 SWD-H checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if 'model' not in ckpt:
            raise KeyError(f"Checkpoint has no 'model' state dict: {ckpt_path}")
        checkpoint_state = ckpt['model']
        model_state = model.state_dict()
        matched_keys = set(checkpoint_state) & set(model_state)
        required_components = {
            "LoRA": "lora_",
            "video projection": "affectgpt_proj",
            "audio projection": "audio_llama_proj",
            "multimodal projection": "multi_llama_proj",
        }
        missing_components = [
            name for name, token in required_components.items()
            if not any(token in key for key in matched_keys)
        ]
        if missing_components:
            raise RuntimeError(
                "The Stage 1 checkpoint is incompatible; no matching weights for: "
                + ", ".join(missing_components)
            )

        load_result = model.load_state_dict(checkpoint_state, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        print("Checkpoint loaded successfully.")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            print("  Missing key sample: " + ", ".join(missing[:20]))
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            print("  Unexpected key sample: " + ", ".join(unexpected[:20]))
        disallowed_unexpected = [
            key for key in unexpected if not key.startswith("ot_hidden_proj.")
        ]
        if disallowed_unexpected:
            raise RuntimeError(
                "Unexpected non-SWD-H checkpoint keys: "
                + ", ".join(disallowed_unexpected[:20])
            )
    else:
        raise FileNotFoundError(
            "Stage 1 SWD-H checkpoint not found. Set model.grpo.ckpt in the "
            f"config or pass --options model.grpo.ckpt=/path/to/checkpoint.pth (got: {ckpt_path!r})."
        )

    model = model.to(device)
    model.train()

    # ====== Reference model (frozen copy for KL regularization) ======
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model = ref_model.to(device)

    # ====== Chat interfaces for generation ======
    model_cfg = cfg.model_cfg
    chat = Chat(model, model_cfg, device=str(device))
    ref_chat = Chat(ref_model, model_cfg, device=str(device))

    # ====== GRPO hyperparameters from config ======
    num_completions = grpo_cfg.get("num_completions", 4)
    max_new_tokens = grpo_cfg.get("max_new_tokens", 80)
    gen_temperature = grpo_cfg.get("gen_temperature", 1.0)
    gen_top_p = grpo_cfg.get("gen_top_p", 0.9)
    clip_eps = grpo_cfg.get("clip_eps", 0.2)
    kl_coef = grpo_cfg.get("kl_coef", 0.01)
    lr = grpo_cfg.get("lr", 1e-5)
    num_epochs = grpo_cfg.get("num_epochs", 3)
    log_interval = grpo_cfg.get("log_interval", 10)
    save_interval = int(grpo_cfg.get("save_interval", 500))
    save_steps_raw = grpo_cfg.get("save_steps", [])
    if isinstance(save_steps_raw, (int, str)):
        save_steps_raw = [save_steps_raw]
    save_steps = {int(step) for step in save_steps_raw}
    max_steps = grpo_cfg.get("max_steps", 5000)
    grad_accum = grpo_cfg.get("grad_accum", 4)

    w_format = grpo_cfg.get("w_format", 3.0)
    w_brevity = grpo_cfg.get("w_brevity", 1.5)
    w_no_reason = grpo_cfg.get("w_no_reason", 2.0)
    w_clean_stop = grpo_cfg.get("w_clean_stop", 1.5)
    w_emotion = grpo_cfg.get("w_emotion", 1.0)
    log_generated_text = grpo_cfg.get("log_generated_text", False)

    output_dir = cfg.run_cfg.get("output_dir", "output_grpo")
    os.makedirs(output_dir, exist_ok=True)

    face_or_frame = 'multiface_audio_face_text'
    for ds_name in cfg.datasets_cfg:
        if hasattr(cfg.datasets_cfg[ds_name], 'face_or_frame'):
            face_or_frame = cfg.datasets_cfg[ds_name].face_or_frame
            break

    # ====== Optimizer (only LoRA + projection params) ======
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # ====== Load data annotations ======
    annotations, wav_root, face_root = load_training_annotations()
    print(f"\nTotal training annotations: {len(annotations)}")

    # ====== Visual processor for face transforms ======
    vis_processor_cfg = model_cfg.get("vis_processor", None)
    if vis_processor_cfg is not None:
        from my_affectgpt.processors import AlproVideoTrainProcessor
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
    else:
        from my_affectgpt.processors import BaseProcessor
        vis_processor = BaseProcessor()

    print(f"\n{'='*50}")
    print(f"GRPO Training Configuration")
    print(f"{'='*50}")
    print(f"  Stage 1 checkpoint: {ckpt_path}")
    print(f"  Completions/sample: {num_completions}")
    print(f"  Max new tokens:     {max_new_tokens}")
    print(f"  Gen temperature:    {gen_temperature}")
    print(f"  Gen top_p:          {gen_top_p}")
    print(f"  Clip epsilon:       {clip_eps}")
    print(f"  KL coefficient:     {kl_coef}")
    print(f"  Learning rate:      {lr}")
    print(f"  Epochs:             {num_epochs}")
    print(f"  Max steps:          {max_steps}")
    print(f"  Save interval:      {save_interval}")
    print(f"  Extra save steps:   {sorted(save_steps)}")
    print(f"  Grad accumulation:  {grad_accum}")
    print(f"  Face or frame:      {face_or_frame}")
    print(f"  Reward weights:     format={w_format}, brevity={w_brevity}, "
          f"no_reason={w_no_reason}, clean_stop={w_clean_stop}, emotion={w_emotion}")
    print(f"  Trainable params:   {sum(p.numel() for p in trainable_params):,}")
    print(f"  Output dir:         {output_dir}")
    print(f"{'='*50}\n")

    autocast_dtype = torch.bfloat16
    user_message = "Please recognize all possible emotional states of the character."

    # ====== Training loop ======
    global_step = 0
    running_reward = 0.0
    running_loss = 0.0
    running_format_hits = 0
    running_total_completions = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        random.shuffle(annotations)

        for sample_idx, sample in enumerate(annotations):
            if global_step >= max_steps:
                break

            name = sample['name']
            subtitle = sample['subtitle']
            gt_emotions = sample['ovlabel']

            try:
                sample_data = load_sample_data(sample, wav_root, face_root,
                                               model_cfg, vis_processor)
            except Exception as e:
                print(f"  Skip sample {sample_idx}: load error - {e}")
                continue

            # Encode multimodal features via Chat
            try:
                audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
                face_hiddens, face_llms = chat.postprocess_face(sample_data)
                multi_hiddens, multi_llms = None, None
                if face_or_frame.startswith('multiface') and face_hiddens is not None and audio_hiddens is not None:
                    _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)

                img_list = {
                    'audio': audio_llms, 'frame': None,
                    'face': face_llms, 'image': None,
                    'multi': multi_llms,
                }
            except Exception as e:
                print(f"  Skip sample {sample_idx}: encode error - {e}")
                continue

            # Build prompt
            prompt = build_prompt(face_or_frame, subtitle, user_message)

            # ====== Step 1: Generate K completions ======
            model.eval()
            try:
                completions, full_decoded_list, token_counts = generate_completions(
                    chat, prompt, img_list,
                    num_completions=num_completions,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_temperature,
                    top_p=gen_top_p)
            except Exception as e:
                print(f"  Skip sample {sample_idx}: generation error - {e}")
                continue

            # ====== Step 2: Compute rewards on FULL decoded output ======
            # full_decoded contains everything the model generated (no ### truncation)
            # This is critical: reward must see reasoning text to penalize it
            rewards = []
            reward_components = []
            for fd, n_tok in zip(full_decoded_list, token_counts):
                # Strip special tokens but keep all generated content.
                raw = fd
                if chat.tokenizer.bos_token and raw.find(chat.tokenizer.bos_token) != -1:
                    raw = raw.split(chat.tokenizer.bos_token)[-1]
                if chat.tokenizer.eos_token and raw.find(chat.tokenizer.eos_token) != -1:
                    raw = raw.split(chat.tokenizer.eos_token)[0]
                raw = raw.split('Assistant:')[-1].strip()
                raw = raw.replace('###', ' ').strip()

                r, comps = compute_reward_with_components(
                    raw, gt_emotions,
                    n_generated_tokens=n_tok,
                    w_format=w_format, w_brevity=w_brevity,
                    w_no_reason=w_no_reason,
                    w_clean_stop=w_clean_stop,
                    w_emotion=w_emotion)
                rewards.append(r)
                reward_components.append(comps)
            rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

            # ====== Step 3: Group-relative advantages ======
            if rewards_tensor.std() < 1e-8:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            if advantages.abs().max() < 1e-6:
                continue

            # ====== Step 4: Compute log probs and GRPO loss ======
            model.train()
            tokenizer = model.llama_tokenizer

            try:
                with torch.no_grad():
                    input_embeds, attention_mask = build_input_embeds(
                        model, chat, prompt, img_list, device=device)

                policy_log_probs_list = []
                ref_log_probs_list = []

                for comp in completions:
                    comp_text = comp + "###"
                    comp_ids = tokenizer(
                        comp_text, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(device)

                    p_log_prob = compute_sequence_log_probs(
                        model, input_embeds, attention_mask,
                        comp_ids, autocast_dtype)
                    policy_log_probs_list.append(p_log_prob)

                    with torch.no_grad():
                        ref_input_embeds, ref_attention_mask = build_input_embeds(
                            ref_model, ref_chat, prompt, img_list, device=device)
                        r_log_prob = compute_sequence_log_probs(
                            ref_model, ref_input_embeds, ref_attention_mask,
                            comp_ids, autocast_dtype)
                        ref_log_probs_list.append(r_log_prob)

                policy_log_probs = torch.stack(policy_log_probs_list)
                ref_log_probs = torch.stack(ref_log_probs_list)

                loss, loss_stats = grpo_loss(policy_log_probs, ref_log_probs, advantages,
                                             clip_eps=clip_eps, kl_coef=kl_coef)
                loss = loss / grad_accum
                loss.backward()

            except Exception as e:
                print(f"  Skip sample {sample_idx}: loss computation error - {e}")
                optimizer.zero_grad()
                continue

            # ====== Gradient accumulation & step ======
            if (sample_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Tracking — format_rate uses full_decoded (not truncated)
                running_reward += rewards_tensor.mean().item()
                running_loss += loss.item() * grad_accum
                for fd in full_decoded_list:
                    raw = fd.split('Assistant:')[-1].strip().replace('###', ' ').strip()
                    if reward_format(raw) == 1.0:
                        running_format_hits += 1
                    running_total_completions += 1

                if global_step % log_interval == 0:
                    avg_reward = running_reward / log_interval
                    avg_loss = running_loss / log_interval
                    avg_format = running_format_hits / max(running_total_completions, 1)
                    best_idx = rewards_tensor.argmax().item()
                    worst_idx = rewards_tensor.argmin().item()
                    avg_tokens = np.mean(token_counts)
                    print(f"  Step {global_step}: loss={avg_loss:.4f}, "
                          f"reward={avg_reward:.3f}, format_rate={avg_format:.2%}, "
                          f"avg_gen_tokens={avg_tokens:.0f}")
                    if log_generated_text:
                        print(f"    [BEST]  processed: '{completions[best_idx][:120]}'")
                        print(f"    [WORST] full_raw:  '{full_decoded_list[worst_idx][:200]}'")
                    print(f"    [DEBUG] token_counts: {token_counts}, "
                          f"rewards: [{', '.join(f'{r:.2f}' for r in rewards)}]")
                    if reward_components:
                        comp_means = {
                            key: float(np.mean([c.get(key, 0.0) for c in reward_components]))
                            for key in reward_components[0]
                        }
                        print("    [REWARD] " + ", ".join(
                            f"{key}={value:.3f}" for key, value in sorted(comp_means.items())))
                    print(f"    [POLICY] log_ratio_mean={loss_stats['log_ratio_mean']:.3f}, "
                          f"log_ratio_absmax={loss_stats['log_ratio_absmax']:.3f}, "
                          f"approx_kl={loss_stats['approx_kl']:.3f}")
                    running_reward = 0.0
                    running_loss = 0.0
                    running_format_hits = 0
                    running_total_completions = 0

                if ((save_interval > 0 and global_step % save_interval == 0)
                        or global_step in save_steps):
                    save_path = os.path.join(output_dir,
                        f"checkpoint_{global_step:06d}_grpo.pth")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                    }, save_path)
                    print(f"  Saved: {save_path}")

        if global_step >= max_steps:
            break

    # Final save
    save_path = os.path.join(output_dir, f"checkpoint_final_{global_step:06d}_grpo.pth")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
    }, save_path)
    print(f"\nGRPO training complete!")
    print(f"  Final checkpoint: {save_path}")
    print(f"  Total steps: {global_step}")


if __name__ == "__main__":
    main()
