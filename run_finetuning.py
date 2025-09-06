# run_finetuning.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def run_training():
    """
    Loads the -base model and dataset, then starts the fine-tuning process.
    """
    # Load Model and Tokenizer
    model_id = "google/flan-t5-base"
    print(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Load and Prepare the Dataset
    dataset_path = "finetune_dataset.jsonl"
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def preprocess_function(examples):
        """Prepares the data into the format required by the model."""
        inputs = [f"Instruction: {instruction}\nResponse:" for instruction in examples["instruction"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        
        labels = tokenizer(text_target=examples["response"], max_length=512, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Define Training Arguments
    # Make sure this matches your HF username and desired model name
    hub_model_id = "IITKProject/iitk-companion-flan-t5-base"
    
    training_args = TrainingArguments(
        output_dir="./iitk_companion_trainer_base",
        num_train_epochs=3,
        per_device_train_batch_size=2, # Using a safe batch size to avoid memory issues
        learning_rate=3e-4,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        logging_dir='./logs_base',
        logging_steps=10,
    )

    # Create Trainer and Start Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting the fine-tuning process. Please let this run to completion.")
    trainer.train()
    print("Fine-tuning complete!")

    # Save the final model AND tokenizer locally
    print("Saving the final model and tokenizer locally...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Push everything to the Hub
    print("Uploading all files to the Hub...")
    trainer.push_to_hub()
    print(f"Model and tokenizer successfully pushed to the Hub at: https://huggingface.co/{hub_model_id}")

if __name__ == "__main__":
    run_training()