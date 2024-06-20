# -*- coding: utf-8 -*-
# @Date: 2024/6/20
# @Author: zhongchao
# @FileName: ch7_utils.py
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

model_name = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size = 64

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
tokenzier = AutoTokenizer.from_pretrained(model_name, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name, language=language, task=task)

def prepare_dataset(batch):
    audio = batch['audio']
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = tokenzier(batch['sentence']).input_ids
    return batch
