## 大模型微调训练营作业总结

### 第三章
1. 情感分析任务
```python
pipe = pipeline(
    task='sentiment-analysis',
    model='lxyuan/distilbert-base-multilingual-cased-sentiments-student',
    return_all_scores=True
)
```
2. 命名实体识别任务
```python
model_name = 'dslim/bert-base-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(task='ner', model=model, tokenizer=tokenizer)
```

### 第四章
1. 使用完整的 YelpReviewFull 数据集训练，对比看 Acc 最高能到多少。
```python
dataset = load_dataset('yelp_review_full')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
metric = evaluate.load('accuracy')

training_args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=20,
    num_train_epochs=3,
    logging_steps=30,
    save_steps=500,
    save_total_limit=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metric,
)

trainer.train(resume_from_checkpoint=True)
trainer.evaluate(test_dataset)
```
output: 
```json
{'eval_loss': 1.0953632593154907,
 'eval_accuracy': 0.63,
 'eval_runtime': 2.9788,
 'eval_samples_per_second': 33.571,
 'eval_steps_per_second': 4.364,
 'epoch': 3.0}
```
2. 加载本地保存的模型，进行评估和再训练更高的 F1 Score。
```python
dataset = load_dataset('squad_v2' if squad_v2 else 'squad')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')
metric = load_metric('squad_v2' if squad_v2 else 'squad')
metric.compute(predictions=formatted_predictions, references=references)
```
output: 
```json
{'exact_match': 75.55345316934721, 'f1': 84.2860083518279}
```

### 第七章
1. 在“LoRA 低秩适配 OpenAI Whisper-Large-V2 语音识别任务”中，为中文语料的训练过程增加过程评估，观察 Train Loss 和 Validation Loss 变化。
```python
dataset_name = "mozilla-foundation/common_voice_11_0"
model_name = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"     # 转录任务

common_voice = DatasetDict()
common_voice['train'] = load_dataset(dataset_name, language_abbr, split='train', trust_remote_code=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name, language=language, task=task)
# 调整音频的采样率为16kHz
common_voice = common_voice.cast_column('audio', Audio(sampling_rate=16000))

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
model.config.forced_decoder_ids = None

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=4,
    lora_alpha=64,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none'
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 1,966,080 || all params: 1,545,271,040 || trainable%: 0.12723204856023188
```
2. 在“LoRA 低秩适配 OpenAI Whisper-Large-V2 语音识别任务”中，当 LoRA 模型训练完成后，使用测试集进行完整的模型评估
```python
peft_config = PeftConfig.from_pretrained(model_dir)
base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    peft_config.base_model_name_or_path,
    load_in_8bit=True,
    device_map='auto'
)
base_model.requires_grad_(False)
peft_model = PeftModel.from_pretrained(base_model, model_dir)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)

metric = evaluate.load('wer')
wer = 100 * metric.compute()
print(f'{wer=:.3f}%')
# wer=65.281%
```

### 第八章
1. 使用 GPTQ 量化 OPT-6.7B 模型。
```python
model_name = 'facebook/opt-6.7b'
gptq_model_dir = 'models/opt-6.7b-gptq'

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset='wikitext2',    # gptq量化需要指定数据集
    desc_act=False
)
gptq_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map='auto'
)

gptq_model.save_pretrained(gptq_model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "Merry Chrismas! I'm glad to"
inputs = tokenizer(text, return_tensors='pt').to(0)
out = gptq_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
# Merry Chrismas! I'm glad to see you're still around.
# I'm still around, just not as much as I used to be.
```
2. 使用 AWQ 量化 Facebook OPT-6.7B 模型。
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig, AutoModelForCausalLM

model_name = 'facebook/opt-6.7b'
awq_model_dir = 'models/opt-6.7b-awq'

model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenzier = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

quant_config = {
    'zero_point': True,
    'q_group_size': 128,
    'w_bit': 4,
    'version': 'GEMM'
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "Merry Chrismas! I'm glad to"
model.quantize(tokenizer, quant_config)

awq_config = AwqConfig(
    bits=quant_config['w_bit'],
    group_size=quant_config['q_group_size'],
    zero_point=quant_config['zero_point'],
    version=quant_config['version'].lower(),
    backend='autoawq'
)

model.model.config.quantization_config = awq_config
model.save_quantized(awq_model_dir)
tokenzier.save_pretrained(awq_model_dir)

awq_tokenizer = AutoTokenizer.from_pretrained(awq_model_dir)
awq_model = AutoModelForCausalLM.from_pretrained(awq_model_dir, device_map='cuda').to(0)

inputs = awq_tokenizer(text, return_tensors='pt').to(0)
out = awq_model.generate(**inputs, max_new_tokens=64)
awq_tokenizer.batch_decode(out[:3], skip_special_tokens=True)
# ["Merry Chrismas! I'm glad to hear you guys are still hanging out, hope it continues to be fun and positive. Keep us updated!\n>I'm glad to hear you guys are still hanging out  Oh yeah, I don't care at all. I get paid either way."]
```

### 第九章
1. 根据硬件资源情况，在 AdvertiseGen 数据集上使用 QLoRA 微调 ChatGLM3-6B 至少 10K examples
```python
model_name = 'THUDM/chatglm3-6b'
model_local_dir = '/root/huggingface/hub/chatglm3-6b'
train_data_path = 'HasturOfficial/adgen'
lora_rank = 4
lora_alpha = 32
lora_dropout = 0.05

dataset = load_dataset(train_data_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_local_dir,
    trust_remote_code=True,
    # revision='b098244'
)
tokenized_dataset = dataset['train'].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,
    remove_columns=column_names
)

tokenized_dataset = tokenized_dataset.shuffle(seed=seed).select(range(12000))
tokenized_dataset = tokenized_dataset.flatten_indices()

data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

q_config = BitsAndBytesConfig(
    load_in_4bit=True,                                      # 4bit量化
    bnb_4bit_quant_type='nf4',                              # nf4 量化
    bnb_4bit_use_double_quant=True,                         # 双量化
    bnb_4bit_compute_dtype=torch.bfloat16,                  # bf16混合精度计算
)

model = AutoModel.from_pretrained(
    model_local_dir,
    quantization_config=q_config,
    device_map='auto',
    trust_remote_code=True,
    # revision='b098244',
)
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 * 1024)
print(f'{memory_footprint_mib:.2f}MiB')
# 3739.69MiB

kbit_model = prepare_model_for_kbit_training(model)
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']  
# ['query_key_value']

lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM,
)

qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()
# trainable params: 974,848 || all params: 6,244,558,848 || trainable%: 0.01561115883009451

training_args = TrainingArguments(
    output_dir=f'models/{model_name}',
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    # per_device_eval_batch_size=8,
    learning_rate=1e-3,
    num_train_epochs=1,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    logging_steps=100,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=2,
    # evaluation_strategy='steps',
    # eval_steps=500,
    optim='adamw_torch',
    fp16=True,
)
trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()
```
2. 推理和对比
```python
model_name = 'THUDM/chatglm3-6b'
peft_model_path = f'models/{model_name}'
peft_config = PeftConfig.from_pretrained(peft_model_path)
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32
)

base_model = AutoModel.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=q_config,
    trust_remote_code=True,
    device_map='auto'
)

base_model.requires_grad_(False)

input_text = '类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领'
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)

response, history = base_model.chat(tokenizer, input_text)
print(f'ChatGLM3-6B 微调前：\n{response}')

peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
response, history = peft_model.chat(tokenizer, input_text)
print(f'ChatGLM3-6B 微调后：\n{response}')
```

### 第十四章 
1. LLaMA 2 指令微调（Alpaca-Style on Dolly-15K Dataset)
示例代码关键训练要素：
- 使用 Dolly-15K 数据集，以 Alpaca 指令风格生成训练数据
- 以 4-bit（NF4）量化精度加载 `LLaMA 2-7B` 模型
- 使用 QLoRA 以 `bf16` 混合精度训练模型
- 使用 `HuggingFace TRL` 的 `SFTTrainer` 实现监督指令微调
- 使用 Flash Attention 快速注意力机制加速训练（需硬件支持）

以 Alpaca-Style 格式化指令数据  
`Alpaca-style` 格式：https://github.com/tatsu-lab/stanford_alpaca#data-release

```python
dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
```

### 第十五章
1. 调整 ZeRO-3 配置文件，使其支持 T5-3B 甚至 T5-11B 模型训练。
```bash
deepspeed --num_gpus=1 translation/run_translation.py \
--deepspeed config/ds_config_zero3.debug.json \
--model_name_or_path t5-3b --per_device_train_batch_size 32 \
--output_dir tmp/t5-3b --overwrite_output_dir --bf16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro \
--source_prefix "translate English to Romanian: " \
--learning_rate 1e-5 \
--max_grad_norm 1
```
