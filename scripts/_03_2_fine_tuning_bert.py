import os
import numpy as np
from datasets import Dataset
from transformers import (
                        AutoTokenizer, AutoModelForTokenClassification,
                        Trainer, TrainingArguments, DataCollatorForTokenClassification
                        )
from seqeval.metrics import classification_report

class bert_Amharic_NER_fine_tuner:
    """
    A class to fine-tune the rasyosef/bert-tiny-amharic for Amharic NER.

    Args:
        conll_path (str): Path to the CoNLL format dataset file.
        model_checkpoint (str, optional): The model checkpoint to use.
            Defaults to "rasyosef/bert-tiny-amharic".
        output_dir (str, optional): The output directory for the fine-tuned model.
            Defaults to None.
    """
    def __init__(self, conll_path,  model_checkpoint = "rasyosef/bert-tiny-amharic",
                output_dir = None):
        print("Initialising bert_Amharic_NER_fine_tuner...")
        self.output_dir  = output_dir
        self.conll_path  =  conll_path
        self.rel_conll_path = os.path.relpath(self.conll_path, os.getcwd())
        self. model_checkpoint =  model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained( model_checkpoint)
        self.model = None
        self.trainer = None
        self.label2id = {}
        self.id2label = {}
        print("\nInitialisation complete.")

    def load_conll_data(self):
        """
        Loads data from a CoNLL format file.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'tokens' and 'ner_tags'.
        """
        print(f"\nLoading data from {self.rel_conll_path}...")
        examples = []
        tokens, tags = [], []
        with open(self.conll_path, encoding = "utf-8") as f:
            for line in f:
                if not line.strip():
                    if tokens:
                        examples.append({'tokens': tokens, 'ner_tags': tags})
                        tokens, tags = [], []
                else:
                    splits = line.strip().split()
                    if len(splits) >= 2:
                        tokens.append(splits[0])
                        tags.append(splits[-1])
        print(f"\nLoaded {len(examples)} examples.")
        return examples

    def tokenize_align(self, example):
        """
        Tokenizes and aligns the tokens with their corresponding NER tags.

        Args:
            example (dict): A dictionary containing 'tokens' and 'labels' (numeric representation of NER tags).
        Returns:
            dict: A dictionary with tokenized input IDs and aligned labels.
        """
        tokenized = self.tokenizer(example["tokens"], 
                                truncation = True, 
                                is_split_into_words = True)
        word_ids = tokenized.word_ids()
        labels = []
        prev = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev:
                labels.append(example["labels"][word_id])
            else:
                labels.append(example["labels"][word_id])
            prev = word_id
        tokenized["labels"] = labels
        return tokenized

    def compute_metrics(self, p):
            """
            Computes evaluation metrics.

            Args:
                p (EvalPrediction): The evaluation prediction object from the Trainer.
            Returns:
                dict: A dictionary containing the computed metrics.
            """
            preds = np.argmax(p.predictions, axis = -1)
            true_preds, true_labels = [], []
            for pred, label in zip(preds, p.label_ids):
                pred_labels = [self.id2label[p] for (p, l) in zip(pred, label) if l != -100]
                true_label_seq = [self.id2label[l] for l in label if l != -100]
                true_preds.append(pred_labels)
                true_labels.append(true_label_seq)
            report = classification_report(true_labels, true_preds, 
                                            output_dict = True, zero_division = 0)

            return {
                **report.get("overall", {}),
                **{f"{label}": scores for label, scores in report.items() if label not in [
                    "macro avg", "weighted avg", "micro avg", "overall"]},
                    "micro avg": report.get("micro avg", {}),
                    "macro avg": report.get("macro avg", {}),
                    "weighted avg": report.get("weighted avg", {})
                    }


    def train(self, epochs = 15):
            """
            Trains the model on the provided CoNLL data.

            Args:
                epochs (int, optional): The number of training epochs. Defaults to 5.
            """
            print("\nLoading and preparing data...")
            data = self.load_conll_data()
            tags = sorted(set(tag for ex in data for tag in ex["ner_tags"]))
            self.label2id = {t: i for i, t in enumerate(tags)}
            self.id2label = {i: t for t, i in self.label2id.items()}
            print("\nData loaded and processed.")

            hf_dataset = Dataset.from_list([
                {"tokens": ex["tokens"], "labels": [self.label2id[t] for t in ex["ner_tags"]]}
                for ex in data
            ]).train_test_split(test_size = 0.2, seed = 42)
            print("\nDataset created and split into train/test.")

            tokenized_ds = hf_dataset.map(self.tokenize_align, batched=False)
            print("\nDataset tokenized and aligned.")

            print("\nSetting up Trainer...")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self. model_checkpoint,
                num_labels = len(self.label2id),
                id2label = self.id2label,
                label2id = self.label2id,
                ignore_mismatched_sizes = True
            )
            print("\nModel loaded and configured.")

            args = TrainingArguments(
                output_dir = self.output_dir,
                eval_strategy = "epoch",
                save_strategy = "epoch",
                logging_strategy = "epoch",
                logging_dir = os.path.join(self.output_dir or "./", "logs"),
                learning_rate = 2e-5,
                per_device_train_batch_size = 8,
                per_device_eval_batch_size = 8,
                num_train_epochs = epochs,
                weight_decay = 0.01,
                report_to = "none", 
                label_smoothing_factor = 0.1 
            )
            print("\nTraining arguments set.")

            self.trainer = Trainer(
                model = self.model,
                processing_class  = self.tokenizer,
                args = args,
                train_dataset = tokenized_ds["train"],
                eval_dataset = tokenized_ds["test"],
                data_collator = DataCollatorForTokenClassification(self.tokenizer),
                compute_metrics = self.compute_metrics
            )
            print("\nTrainer initialised.")

            print("\nStarting training...")
            self.trainer.train()
            print("Training complete.")

            print("\nSaving model...")
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print("Model saved.")

            print("\nTraining complete and model saved.")

    def evaluate(self):
        """
        Evaluates the trained model and prints a formatted summary.

        Returns:
            dict: Raw evaluation metrics.
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialised. Call .train() first.")

        print("\nEvaluating model...")
        metrics = self.trainer.evaluate()
        print("\nEvaluation complete.")

        # Format and print key metrics
        if "eval_loss" in metrics:
            print(f"\nEval Loss: {metrics['eval_loss']:.4f}")

        print("Per-Entity Performance:")
        for label, scores in metrics.items():
            if label.startswith("eval_") and isinstance(scores, dict) and "precision" in scores:
                name = label.replace("eval_", "")
                p = scores["precision"] * 100
                r = scores["recall"] * 100
                f1 = scores["f1-score"] * 100
                support = scores["support"]
                print(f"  • {name:<10} | P: {p:5.1f}%  R: {r:5.1f}%  F1: {f1:5.1f}%  (Support: {support})")

        print("\n📦 Runtime Info:")
        print(f"  - Runtime: {metrics.get('eval_runtime', 0):.2f}s")
        print(f"  - Samples/sec: {metrics.get('eval_samples_per_second', 0):.1f}")
        print(f"  - Epoch: {metrics.get('epoch', 0)}")

        return metrics
