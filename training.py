from transformers import TrainingArguments
from transformers import Trainer, TrainingArguments, AdamW
from model_configuration import *
from transformers import Trainer
training_args = TrainingArguments(
    output_dir="./results",         
    num_train_epochs=1,             
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,    
    learning_rate=5e-05,            
    weight_decay=0.01,              
    logging_dir="./logs",           
    logging_steps=10,                
    seed=42,                       
    evaluation_strategy="steps",    
    eval_steps=10,                   
    warmup_steps=int(0.1 * 20),      
    optim="adamw_torch",          
    lr_scheduler_type="linear",      
    fp16=True,                       
)

optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
# Define the trainer
trainer = Trainer(
    model=model,                      
    args=training_args,              
    train_dataset=shuffled_dataset["train"],      
    eval_dataset=shuffled_dataset["test"],       
    optimizers=(optimizer, None),     
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()