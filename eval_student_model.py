#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for student model using AffectGPT's evaluation functions.
"""
import os
import sys
import glob
import numpy as np
import argparse

# Import AffectGPT evaluation functions
from my_affectgpt.evaluation.wheel import *
from my_affectgpt.evaluation.ew_metric import *

# Import dataset classes
from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
from my_affectgpt.datasets.datasets.meld import MELD_Dataset
from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset
from my_affectgpt.datasets.datasets.cmumosi import CMUMOSI_Dataset
from my_affectgpt.datasets.datasets.cmumosei import CMUMOSEI_Dataset
from my_affectgpt.datasets.datasets.sims import SIMS_Dataset
from my_affectgpt.datasets.datasets.simsv2 import SIMSv2_Dataset
from my_affectgpt.datasets.datasets.ovmerdplus_dataset import OVMERDPlus_Dataset

# Import config
import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


def func_read_batch_calling_model(modelname):
    """Load vLLM model for openset label extraction."""
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


def calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    """Calculate discrete emotion scores (hitrate, mscore)."""
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    hitrate, mscore = hitrate_metric_calculation(name2gt=name2gt, openset_npz=openset_npz, inter_print=inter_print)
    return hitrate, mscore


def calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    """Calculate dimension scores (fscore, accuracy)."""
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    
    # Map openset to sentiment
    sentiment_npz = epoch_root[:-4]+'-sentiment.npz'
    if not os.path.exists(sentiment_npz):
        openset_to_sentiment_batchcalling(openset_npz=openset_npz, store_npz=sentiment_npz,
                                          llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    
    # Convert sentiment labels to numerical values
    name2pred = {}
    filenames = np.load(sentiment_npz, allow_pickle=True)['filenames']
    fileitems = np.load(sentiment_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        if item == 'positive':
            name2pred[name] = 1
        elif item == 'negative':
            name2pred[name] = -1
        elif item == 'neutral':
            name2pred[name] = 0
        else:  # Unable to parse label
            if inter_print: print(f'error sample: {name}, {item}')
            name2pred[name] = 0
    
    # Calculate metrics
    val_labels, val_preds = [], []
    for name in name2gt:
        val_labels.append(name2gt[name])
        val_preds.append(name2pred[name])
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    
    non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0])
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
    fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')
    return fscore, accuracy


def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    """Calculate open-vocabulary scores (fscore, precision, recall)."""
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    
    fscore, precision, recall = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return fscore, precision, recall


def get_dataset2cls(dataset):
    """Get dataset class for reading ground truth."""
    if dataset == 'MER2023' or dataset == 'mer2023':
        return MER2023_Dataset()
    if dataset == 'MER2024' or dataset == 'mer2024':
        return MER2024_Dataset()
    if dataset == 'MELD' or dataset == 'meld':
        return MELD_Dataset()
    if dataset == 'IEMOCAPFour' or dataset == 'iemocapfour':
        return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI' or dataset == 'cmumosi':
        return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI' or dataset == 'cmumosei':
        return CMUMOSEI_Dataset()
    if dataset == 'SIMS' or dataset == 'sims':
        return SIMS_Dataset()
    if dataset == 'SIMSv2' or dataset == 'simsv2':
        return SIMSv2_Dataset()
    if dataset == 'OVMERDPlus' or dataset == 'ovmerdplus':
        return OVMERDPlus_Dataset()
    raise ValueError(f'Unknown dataset: {dataset}')


def get_discrete_or_dimension_flag(dataset):
    """Determine dataset type."""
    dataset_upper = dataset.upper()
    discrete_datasets = ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFOUR']
    dimension_datasets = ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSV2']
    ovlabel_datasets = ['OVMERDPLUS']
    
    if dataset_upper in discrete_datasets:
        return 'discrete'
    elif dataset_upper in dimension_datasets:
        return 'dimension'
    elif dataset_upper in ovlabel_datasets:
        return 'ovlabel'
    else:
        return 'unknown'


def get_emo2idx_idx2emo(dataset_cls):
    """Get emotion mapping from dataset class."""
    emo2idx, idx2emo = {}, {}
    
    if hasattr(dataset_cls, 'get_emo2idx_idx2emo'): 
        emo2idx, idx2emo = dataset_cls.get_emo2idx_idx2emo()
        # post process [不同数据集的标签表示有些许差异，进行统一化处理]
        if 'happy' in emo2idx: emo2idx['joy']   = emo2idx['happy']
        if 'anger' in emo2idx: emo2idx['angry'] = emo2idx['anger']
        if 'sad'   in emo2idx: emo2idx['sadness'] = emo2idx['sad']
        if 'joy'   in emo2idx: emo2idx['happy'] = emo2idx['joy']
        if 'angry' in emo2idx: emo2idx['anger'] = emo2idx['angry']
        if 'sadness' in emo2idx: emo2idx['sad'] = emo2idx['sadness']
        
        # do the same process for idx2emo
        for idx in idx2emo:
            emo = idx2emo[idx]
            if emo == 'happy': idx2emo[idx] = 'joy'
            if emo == 'sad':   idx2emo[idx] = 'sadness'
            if emo == 'angry': idx2emo[idx] = 'anger'
    
    return emo2idx, idx2emo


def main_zeroshot_scores(input_dir, debug=False, test_epochs='', inter_print=True):
    """Main evaluation function matching evaluation-scoreonly.py."""
    
    # Read dataset name from path
    # input_dir format: output/results-mer2023/student_clip_qwen3
    if 'results-' in input_dir:
        dataset = input_dir.split('results-')[1].split('/')[0]
        # Convert to proper case (mer2023 -> MER2023)
        for ds_name in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'OVMERDPlus']:
            if ds_name.lower() == dataset.lower():
                dataset = ds_name
                break
    else:
        raise ValueError(f'Cannot parse dataset name from {input_dir}')
    
    disordim_flag = get_discrete_or_dimension_flag(dataset)
    
    if inter_print:
        print(f'process root: {input_dir}')
        print(f'process dataset: {dataset} => {disordim_flag}')
    
    # Get dataset class and ground truth labels
    dataset_cls = get_dataset2cls(dataset)
    name2gt = dataset_cls.get_test_name2gt()
    
    if inter_print:
        print(f'target sample number: {len(name2gt)}')
    
    # Convert discrete labels to string format if needed
    if disordim_flag == 'discrete':
        emo2idx, idx2emo = get_emo2idx_idx2emo(dataset_cls)
        for name in name2gt:
            gt = name2gt[name]
            if not isinstance(gt, str) and idx2emo:
                name2gt[name] = idx2emo[gt]
    
    # Load vLLM model
    if not debug:
        llm, tokenizer, sampling_params = func_read_batch_calling_model('Qwen25')
    else:
        llm, tokenizer, sampling_params = None, None, None
    
    # Find result files
    import glob
    npz_files = sorted(glob.glob(f"{input_dir}/*.npz"))
    npz_files = [f for f in npz_files if '-openset' not in f and '-sentiment' not in f]
    
    whole_score1s, whole_score2s, whole_score3s = [], [], []
    
    for epoch_root in npz_files:
        if disordim_flag == 'discrete':
            hitrate, mscore = calculate_discrete_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print:
                print(f'hitrate: {hitrate}')
            whole_score1s.append(hitrate)
            whole_score2s.append(mscore)
            whole_score3s.append(0)
            
        elif disordim_flag == 'dimension':
            fscore, acc = calculate_dimension_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print:
                print(f'fscore: {fscore}, acc: {acc}')
            whole_score1s.append(fscore)
            whole_score2s.append(acc)
            whole_score3s.append(0)
        
        elif disordim_flag == 'ovlabel':
            fscore, precision, recall = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
            if inter_print:
                print(f'fscore: {fscore}, precision: {precision}, recall: {recall}')
            whole_score1s.append(fscore)
            whole_score2s.append(precision)
            whole_score3s.append(recall)

        if inter_print:
            print('=========================')

    # Return best scores
    best_index = np.argmax(whole_score1s)
    best_score1 = whole_score1s[best_index]
    best_score2 = whole_score2s[best_index]
    best_score3 = whole_score3s[best_index]
    
    if disordim_flag == 'discrete':
        if inter_print:
            print(f'{dataset}: best hitrate: {best_score1:.4f}; best mscore: {best_score2:.4f}')
    elif disordim_flag == 'dimension':
        if inter_print:
            print(f'{dataset}: best fscore: {best_score1:.4f}; best acc: {best_score2:.4f}')
    elif disordim_flag == 'ovlabel':
        if inter_print:
            print(f'{dataset}: best fscore: {best_score1:.4f}; best precision: {best_score2:.4f}; best recall: {best_score3:.4f}')
    
    return best_score1, best_score2, best_score3


def main():
    """Evaluate student model on all datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='student_clip_qwen3')
    args = parser.parse_args()
    
    model_name = args.model_name
    datasets = ["mer2023", "mer2024", "meld", "iemocapfour", 
                "cmumosi", "cmumosei", "sims", "simsv2", "ovmerdplus"]
    
    print("=" * 80)
    print(f"Evaluating Student Model: {model_name}")
    print("=" * 80)
    print()
    
    all_scores = []
    
    for dataset in datasets:
        result_dir = f"output/results-{dataset}/{model_name}"
        
        if not os.path.exists(result_dir):
            print(f"⚠️  {dataset.upper()}: Results directory not found")
            print()
            continue
        
        print(f"{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            score1, score2, score3 = main_zeroshot_scores(
                result_dir, 
                debug=False,
                test_epochs='',
                inter_print=True
            )
            
            all_scores.append({
                'dataset': dataset,
                'score1': score1,
                'score2': score2,
                'score3': score3
            })
            
        except Exception as e:
            print(f"⚠️  Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Summary
    if all_scores:
        print("\n" + "=" * 80)
        print("STUDENT MODEL - FINAL RESULTS")
        print("=" * 80)
        
        for result in all_scores:
            print(f"{result['dataset'].upper()}: {result['score1']*100:.2f}%")
        
        avg_score = np.mean([r['score1'] for r in all_scores])
        print(f"\nAVERAGE: {avg_score*100:.2f}%")
        print("=" * 80)
    else:
        print("⚠️  No successful evaluations!")


if __name__ == "__main__":
    main()
