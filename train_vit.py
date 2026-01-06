import numpy as np
import torch
import evaluate  # Replaces load_metric
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# --- 1. Global Definitions ---

# Reusable metric loading
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(p):
    """Calculates accuracy for the evaluation set."""
    return accuracy_metric.compute(
        predictions=np.argmax(p.predictions, axis=1), 
        references=p.label_ids
    )

def transform(example_batch):
    """Preprocesses a batch of images for the ViT model."""
    inputs = processor(
        [x.convert("RGB") for x in example_batch["image"]], 
        return_tensors="pt"
    )
    inputs["labels"] = example_batch["label"]
    return inputs

# Initialize processor globally
model_id = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_id)

# --- 2. The Main Execution Block ---

if __name__ == "__main__":
    # === CHANGE 1: Load from your LOCAL folder ===
    # Point this to the folder containing 'train' and 'val'
    dataset_path = r"C:\Users\Harun\Documents\Web\ProjectNateHiggerson\horse-dataset"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # 'imagefolder' automatically detects the 'train' and 'val' subfolders
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    
    # Check if it loaded correctly
    print("Dataset structure:", dataset)
    # Expected output: DatasetDict({ train: ..., validation: ... })

    # Get labels from the 'train' split
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    print(f"Found breeds: {labels}")

    # === (Removed the manual train_test_split code here) ===

    # 2. Load Model
    print(f"Initializing ViT for {num_labels} horse breeds...")
    model = ViTForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    # 3. Apply Transformations
    dataset = dataset.with_transform(transform)

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./vit-horse",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_num_workers=0, # Keep this 0 for Windows safety
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        
        # === CHANGE 2: Use the 'validation' split detected by imagefolder ===
        eval_dataset=dataset["validation"], 
        
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    # 6. Train and Save
    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./vit-horse-model")
    processor.save_pretrained("./vit-horse-model")

    print("Success! Model saved to ./vit-horse-model")