from peft import LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model
from torch import nn
from transformers import Trainer


peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# not quite sure how to do this...
# my_trainer = Trainer(model=model, train_dataset=)
