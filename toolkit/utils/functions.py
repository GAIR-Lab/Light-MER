import os
import re

import cv2
import numpy as np
import pandas as pd
import torchaudio
from PIL import Image


def string_to_list(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if value == '' or pd.isna(value):
        return []

    value = str(value).strip()
    if value.startswith('['):
        value = value[1:]
    if value.endswith(']'):
        value = value[:-1]
    return [item.strip() for item in re.split('[\'\",]', value)
            if item.strip() not in ['', ',']]


def func_gain_videopath(video_root, vid_name):
    for suffix in ('.mp4', '.avi'):
        candidate = f"{video_root}/{vid_name}{suffix}"
        if os.path.exists(candidate):
            return candidate
    return f"{video_root}/{vid_name}.mp4"


def func_gain_audiopath(video_root, vid_name):
    return f"{video_root}/{vid_name}.wav"


def func_gain_name2trans(trans_path):
    from toolkit.utils.read_files import func_read_key_from_csv

    names = func_read_key_from_csv(trans_path, 'name')
    chis = func_read_key_from_csv(trans_path, 'chinese')
    return {name: chi for name, chi in zip(names, chis)}


def func_read_audio_second(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if len(waveform.shape) == 2:
        return waveform.shape[1] / sr
    if len(waveform.shape) == 1:
        return len(waveform) / sr
    raise ValueError('Unsupported waveform shape')


def func_opencv_to_image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def func_decord_to_image(img):
    return Image.fromarray(img)


def func_opencv_to_decord(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def func_discrte_label_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def func_label_distribution(labels):
    return func_discrte_label_distribution(labels)
