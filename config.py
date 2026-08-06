# *_*coding:utf-8 *_*
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _path_from_env(env_name, default_relative_path):
    path = os.environ.get(env_name)
    if path:
        return os.path.expanduser(path)
    return os.path.join(PROJECT_ROOT, default_relative_path)


def _join(root, *parts):
    return os.path.join(root, *parts)


## Runtime roots. Override these in your shell or with absolute paths in this file.
AFFECTGPT_ROOT = PROJECT_ROOT
MODEL_ROOT = _path_from_env("SWDH_MODEL_ROOT", "models")
DATASET_ROOT = _path_from_env("SWDH_DATASET_ROOT", "dataset")
EMOTION_WHEEL_ROOT = _path_from_env("SWDH_EMOTION_WHEEL_ROOT", "emotion_wheel")
RESULT_ROOT = _path_from_env("SWDH_RESULT_ROOT", "output/results")


#######################
## 所有模型的存储路径
#######################
PATH_TO_LLM = {
    # Used by eval_student_model.py as a vLLM label extractor.
    'Qwen25': _join(MODEL_ROOT, 'Qwen2.5-7B-Instruct'),
    # Teacher LLM and stage1 student LLM.
    'Qwen3': _join(MODEL_ROOT, 'Qwen3-8B'),
    'Qwen3_0.6B': _join(MODEL_ROOT, 'Qwen3-0.6B'),
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': _join(MODEL_ROOT, 'clip-vit-large-patch14'),
    'CLIP_VIT_BASE':  _join(MODEL_ROOT, 'clip-vit-base-patch16'),
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  _join(MODEL_ROOT, 'chinese-hubert-large'),
    'HUBERT_BASE':   _join(MODEL_ROOT, 'chinese-hubert-base'),
}

#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'MER2025OV':      _join(DATASET_ROOT, 'mer2025-dataset'),
    'MERCaptionPlus': _join(DATASET_ROOT, 'mer2025-dataset'),
    'OVMERD':         _join(DATASET_ROOT, 'mer2025-dataset'),
    'MER2023':        _join(DATASET_ROOT, 'mer2023-dataset-process'),
    'MER2024':        _join(DATASET_ROOT, 'mer2024-dataset-process'),
    'IEMOCAPFour':    _join(DATASET_ROOT, 'iemocap-process'),
    'CMUMOSI':        _join(DATASET_ROOT, 'cmumosi-process'),
    'CMUMOSEI':       _join(DATASET_ROOT, 'cmumosei-process'),
    'SIMS':           _join(DATASET_ROOT, 'sims-process'),
    'SIMSv2':         _join(DATASET_ROOT, 'simsv2-process'),
    'MELD':           _join(DATASET_ROOT, 'meld-process'),
    'OVMERDPlus':     _join(DATASET_ROOT, 'ovmerdplus-process'),
}

PATH_TO_RAW_AUDIO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'audio'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'audio'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'audio'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subaudio'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'audio'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'video'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'video'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'video'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo_new'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video_new'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'video'),
}
PATH_TO_RAW_FACE = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'openface_face'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'openface_face'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'openface_face'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'openface_face'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'openface_face'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'openface_face'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'openface_face'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'subtitle_chieng.csv'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'subtitle_chieng.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'subtitle_eng.csv'),
}
PATH_TO_LABEL = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'track2_test.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'xxx'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'xxx'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
    'OVMERDPlus': os.path.join(DATA_DIR['OVMERDPlus'], 'ovlabel.csv'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
