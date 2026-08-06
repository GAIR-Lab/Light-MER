"""
Inference over Light-MER Stage 2 M-GRPO checkpoints.

The recommended public setting uses explicit generation parameters and can
prepare LoRA adapters before generation. Existing result files are skipped so
interrupted sweeps can resume safely.
"""

import os
import time
import glob
import argparse
import hashlib
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import *

import config
from toolkit.utils.read_files import *


def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1
    return list(set(face_or_frame_candidates))[0]


def get_name2cls(dataset):
    if dataset == 'MER2023':     return MER2023_Dataset()
    if dataset == 'MER2024':     return MER2024_Dataset()
    if dataset == 'MELD':        return MELD_Dataset()
    if dataset == 'IEMOCAPFour': return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':     return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':    return CMUMOSEI_Dataset()
    if dataset == 'SIMS':        return SIMS_Dataset()
    if dataset == 'SIMSv2':      return SIMSv2_Dataset()
    if dataset == 'MER2025OV':   return MER2025OV_Dataset()
    if dataset == 'OVMERDPlus':  return OVMERDPlus_Dataset()
    return None


def get_user_message(dataset_cls, zeroshot, outside_user_message):
    if outside_user_message is not None:
        return outside_user_message
    elif zeroshot:
        return dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light-MER Stage 2 M-GRPO inference")
    parser.add_argument("--cfg-path", required=True, help="Path to the Stage 2 config YAML")
    parser.add_argument("--options", nargs="+", help="Override config params")
    parser.add_argument("--ckpt-dir", required=True, help="Directory containing GRPO checkpoints")
    parser.add_argument("--dataset", default='inferenceData')
    parser.add_argument(
        '--zeroshot', action=argparse.BooleanOptionalAction, default=True,
        help='Use the benchmark zero-shot emotion prompt (recommended).')
    parser.add_argument('--outside_user_message', default=None)
    parser.add_argument('--outside_face_or_frame', default=None)
    parser.add_argument('--max-new-tokens', dest='max_new_tokens', type=int, default=1200,
                        help='Maximum new tokens for generation. The released setting uses 1200.')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='Generation temperature. The public Stage 2 setting uses 0.9.')
    parser.add_argument('--top-p', dest='top_p', type=float, default=0.9,
                        help='Nucleus sampling top_p. The released setting uses 0.9.')
    parser.add_argument('--no-sample', action='store_true',
                        help='Disable sampling and use greedy-style generation.')
    parser.add_argument('--repetition-penalty', dest='repetition_penalty', type=float, default=1.0,
                        help='Generation repetition penalty. The released setting uses 1.0.')
    parser.add_argument('--no-repeat-ngram-size', dest='no_repeat_ngram_size', type=int, default=0,
                        help='Block repeated ngrams during generation. The released setting uses 0.')
    parser.add_argument(
        '--quiet-prompt', action=argparse.BooleanOptionalAction, default=True,
        help='Do not print full multimodal prompts (recommended).')
    parser.add_argument(
        '--compact-log', action=argparse.BooleanOptionalAction, default=True,
        help='Do not print subtitle or prediction content (recommended).')
    parser.add_argument('--start-step', type=int, default=None,
                        help='Only process GRPO checkpoints with step >= this value.')
    parser.add_argument('--end-step', type=int, default=None,
                        help='Only process GRPO checkpoints with step <= this value.')
    parser.add_argument('--base-root', default=None,
                        help='Override inference base_root without editing the YAML.')
    parser.add_argument('--run-tag', default='stage2_mgrpo_qwen3_0_6b',
                        help='Leaf directory under results-* used to store npz files.')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even when a target npz already exists.')
    parser.add_argument('--sample-limit', type=int, default=None,
                        help='Only process the first N samples for timing/debug runs.')
    parser.add_argument(
        '--merge-lora-for-inference',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Merge LoRA adapters before inference (recommended).')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 M-GRPO inference requires a CUDA GPU")

    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)

    inference_datasets = ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour',
                          'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'OVMERDPlus']

    # ====== Find all GRPO checkpoints ======
    ckpt_dir = args.ckpt_dir
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_[0-9]*_grpo.pth")))
    if args.start_step is not None or args.end_step is not None:
        filtered = []
        for ckpt in all_ckpts:
            base = os.path.basename(ckpt)
            try:
                step = int(base.split('_')[1])
            except Exception:
                continue
            if args.start_step is not None and step < args.start_step:
                continue
            if args.end_step is not None and step > args.end_step:
                continue
            filtered.append(ckpt)
        all_ckpts = filtered
    print(f"\n====== Found {len(all_ckpts)} GRPO checkpoints in {ckpt_dir} ======")
    for ckpt in all_ckpts:
        print(f"  {os.path.basename(ckpt)}")
    print()

    if len(all_ckpts) == 0:
        raise FileNotFoundError(
            f"No checkpoint_[step]_grpo.pth files found in {ckpt_dir}"
        )

    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print(f"Face or frame: {face_or_frame}")

    if args.dataset == 'inferenceData':
        process_datasets = inference_datasets
    else:
        process_datasets = args.dataset.split(',')
    print(f"Process datasets: {process_datasets}")
    print(f"Generation: temperature={args.temperature}, top_p={args.top_p}, do_sample={not args.no_sample}, "
          f"max_new_tokens={args.max_new_tokens}, repetition_penalty={args.repetition_penalty}, "
          f"no_repeat_ngram_size={args.no_repeat_ngram_size}, quiet_prompt={args.quiet_prompt}, "
          f"compact_log={args.compact_log}, merge_lora_for_inference={args.merge_lora_for_inference}")

    # ====== Build model once ======
    model_cls = registry.get_model_class(model_cfg.arch)
    model = None

    for ckpt_idx, ckpt_path in enumerate(all_ckpts):
        ckpt_name = os.path.basename(ckpt_path)[:-4]  # e.g. checkpoint_000500_grpo

        print(f"\n{'='*60}")
        print(f"[{ckpt_idx+1}/{len(all_ckpts)}] Loading: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        if args.merge_lora_for_inference and model is not None:
            del model
            torch.cuda.empty_cache()
            model = None
        if model is None:
            model = model_cls.from_config(model_cfg)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if 'model' not in ckpt:
            raise KeyError(f"Checkpoint has no 'model' state dict: {ckpt_path}")
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval()
        if args.merge_lora_for_inference:
            if hasattr(model.llama_model, 'merge_and_unload'):
                print('  Merging LoRA adapters into LLM for inference...')
                model.llama_model = model.llama_model.merge_and_unload()
                model = model.to(device).eval()
            else:
                raise RuntimeError(
                    "LoRA merge requested, but llama_model has no merge_and_unload()"
                )
        chat = Chat(model, model_cfg, device=device)

        for dataset in process_datasets:
            print(f"\n  --- Dataset: {dataset} ---")
            dataset_cls = get_name2cls(dataset)
            if dataset_cls is None:
                print(f"  Skipping unknown dataset: {dataset}")
                continue

            dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor")
            img_processor_cfg = inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(
                    vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(
                    img_processor_cfg.train.name).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms

            test_names = dataset_cls.read_test_names()
            if args.sample_limit is not None:
                test_names = test_names[:args.sample_limit]
            name2subtitle = dataset_cls.name2subtitle

            base_root = args.base_root if args.base_root is not None else inference_cfg.base_root
            save_root = os.path.join(base_root + f'-{dataset.lower()}', args.run_tag)
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, f'{ckpt_name}.npz')
            if os.path.exists(save_path) and not args.force:
                print(f"  Already exists, skip: {save_path}")
                continue

            name2reason = {}
            sample_times = []
            dataset_start = time.time()

            for ii, name in enumerate(test_names):
                sample_start = time.time()
                subtitle = name2subtitle[name]
                if not args.compact_log:
                    print(f'  process on {ii}|{len(test_names)}: {name} | {subtitle}')

                sample = {'name': name}
                video_path, image_path, audio_path, face_npy = None, None, None, None
                if hasattr(dataset_cls, '_get_video_path'): video_path = dataset_cls._get_video_path(sample)
                if hasattr(dataset_cls, '_get_audio_path'): audio_path = dataset_cls._get_audio_path(sample)
                if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
                if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)
                sample_data = dataset_cls.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)

                with torch.inference_mode():
                    audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
                    frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                    face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
                    _,             image_llms = chat.postprocess_image(sample_data)
                    multi_llms = None
                    if face_or_frame.startswith('multiface'):
                        _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
                    elif face_or_frame.startswith('multiframe'):
                        _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

                    img_list = {
                        'audio': audio_llms, 'frame': frame_llms,
                        'face': face_llms, 'image': image_llms,
                        'multi': multi_llms,
                    }

                    user_message = get_user_message(dataset_cls, args.zeroshot, args.outside_user_message)
                    prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)

                    response = chat.answer_sample(prompt=prompt, img_list=img_list,
                                                  num_beams=1, temperature=args.temperature,
                                                  do_sample=not args.no_sample, top_p=args.top_p,
                                                  max_new_tokens=args.max_new_tokens, max_length=2000,
                                                  repetition_penalty=args.repetition_penalty,
                                                  no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                  print_prompt=not args.quiet_prompt)
                name2reason[name] = response
                sample_elapsed = time.time() - sample_start
                sample_times.append(sample_elapsed)
                if args.compact_log:
                    sample_ref = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]
                    print(f'  [{sample_elapsed:.2f}s] {dataset}:{sample_ref} chars={len(response)}')
                else:
                    print(f'  [{sample_elapsed:.2f}s] {response}')

            dataset_elapsed = time.time() - dataset_start
            avg_time = np.mean(sample_times) if sample_times else 0
            print(f'\n  ====== Timing: {dataset} | {ckpt_name} ======')
            print(f'    Total: {len(sample_times)} samples, {dataset_elapsed:.1f}s ({dataset_elapsed/60:.1f}min)')
            if sample_times:
                print(f'    Avg: {avg_time:.2f}s, Min: {min(sample_times):.2f}s, Max: {max(sample_times):.2f}s')
            else:
                print('    No samples were processed.')

            print(f'  Saving: {save_path}')
            np.savez_compressed(save_path, name2reason=name2reason)

    print(f"\n{'='*60}")
    print(f"All done! Processed {len(all_ckpts)} checkpoints × {len(process_datasets)} datasets")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
