"""
Training script for dictionary embedding model.

Fine-tunes a sentence transformer on word-definition pairs using contrastive learning
with Matryoshka Representation Learning.
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_and_split_dataset(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Load CSV dataset and split into train/val/test sets."""
    dataset = load_dataset("csv", data_files=data_path)

    test_ratio = 1.0 - train_ratio - val_ratio
    temp_ratio = val_ratio + test_ratio

    # Split into train and temp (val+test)
    splits = dataset['train'].train_test_split(test_size=temp_ratio)
    train_dataset = splits['train']
    temp = splits['test']

    # Split temp into val and test
    val_test_ratio = test_ratio / temp_ratio
    temp_splits = temp.train_test_split(test_size=val_test_ratio)
    val_dataset = temp_splits['train']
    test_dataset = temp_splits['test']

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_evaluator(val_dataset):
    """Create Information Retrieval evaluator for word-definition matching."""
    evaluator = InformationRetrievalEvaluator(
        queries={i: example['word'] for i, example in enumerate(val_dataset)},
        corpus={i: example['definition'] for i, example in enumerate(val_dataset)},
        relevant_docs={i: [i] for i in range(len(val_dataset))},
        name='dictionary-val'
    )
    return evaluator


def main(args):
    # Load base model
    print(f"Loading base model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # Setup Matryoshka loss
    base_loss = losses.MultipleNegativesRankingLoss(model)
    mrl_loss = losses.MatryoshkaLoss(model, base_loss, args.matryoshka_dims)
    print(f"Using Matryoshka dimensions: {args.matryoshka_dims}")

    # Load and split dataset
    print(f"\nLoading dataset from: {args.data_path}")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(args.data_path)

    # Create evaluator
    evaluator = create_evaluator(val_dataset)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
    )

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=mrl_loss,
        args=training_args,
        evaluator=evaluator
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model locally
    print(f"\nSaving model to: {args.model_save_path}")
    os.makedirs(args.model_save_path, exist_ok=True)
    model.save_pretrained(args.model_save_path)

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if args.hf_repo_id is None:
            raise ValueError("--hf_repo_id must be specified when --push_to_hub is set")

        print(f"\nPushing to HuggingFace Hub: {args.hf_repo_id}")
        login(token=os.getenv("HF_TOKEN"))
        model.push_to_hub(repo_id=args.hf_repo_id)
        print("Successfully pushed to Hub!")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dictionary embedding model with contrastive learning"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base sentence transformer model to fine-tune"
    )
    parser.add_argument(
        "--matryoshka_dims",
        type=int,
        nargs="+",
        default=[384, 256],
        help="Target dimensions for Matryoshka loss"
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/CSW24defs.csv",
        help="Path to CSV file with word-definition pairs"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for training outputs and checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluation frequency in steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Checkpoint save frequency in steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Logging frequency in steps"
    )

    # Save and push arguments
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/dict-embed",
        help="Local path to save final model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g., 'username/model-name')"
    )

    args = parser.parse_args()
    main(args)
