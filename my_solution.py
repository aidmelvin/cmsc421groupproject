from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict, Dataset


device = "cuda"
model_name_or_path = "facebook/opt-1.3b"
tokenizer_name_or_path = "facebook/opt-1.3b"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-3
num_epochs = 3
batch_size = 8

# creating model
# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,  
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.01,
    bias="none",
)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

# data = load_dataset("Abirate/english_quotes")
data = load_dataset("aidmelvin/my_llm_test")
# data = my_data
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
data = data.map(lambda samples: tokenizer(samples["sentence"]), batched=True)
model = model.to(device)

##############################################################################################
########################## inferencing before fine-tuning ####################################
##############################################################################################

# batch = tokenizer("Two things are infinite: ", return_tensors="pt").input_ids.to(device)

# model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# with torch.cuda.amp.autocast():
#     # output_tokens = model.generate(**batch, max_new_tokens=50)
#     output_tokens = model.generate(input_ids=batch, max_new_tokens=50)

# print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))

##############################################################################################

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=20,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

batch = tokenizer("What is the purpose of the universe?: ", return_tensors="pt").input_ids.to(device)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# print(model.eval())

with torch.cuda.amp.autocast():
    # output_tokens = model.generate(**batch, max_new_tokens=50)
    output_tokens = model.generate(input_ids=batch, max_new_tokens=10)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
