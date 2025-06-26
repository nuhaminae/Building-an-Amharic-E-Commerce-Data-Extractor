import csv
import os
from scripts._03_1_fine_tuning_afroxlmr import afroxlmrAmharicNERFineTuner
from scripts._04_1_model_interpretability import predict_with_logits
from transformers import (AutoTokenizer, 
                            AutoModelForTokenClassification)

class EnrichCSVwithNER:
    """
    A utility class for applying a fine-tuned Amharic NER model 
    to a raw CSV and saving an enriched version with extracted entities.
    """
    def __init__(self, output_dir, conll_path,
                    csv_path, output_path = None):
        """
        Initialises the class.

        Args:
            output_dir (str): Directory where the fine-tuned model is stored or to be saved.
            conll_path (str): Path to training data if model training is needed.
            csv_path (str): Path to the input CSV file with scraped vendor messages.
            output_path (str): Path to save the enriched CSV.
        """
        print("\nInitialising NER Enricher...")
        self.ner = afroxlmrAmharicNERFineTuner(conll_path, output_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        self.id2label = self.model.config.id2label

        self.csv_path = csv_path
        self.output_folder = output_path

        if self.model:
            print (f"\nNER model config loaded with labels:{self.id2label }")

        if self.model is None:
                    print("\nNER model not loaded. Consider training or loading weights.")

    def extract_entities(self, text):
        """
        Applies the NER model to a given text and extracts product and price entities.

        Args:
            text (str): The input message to analyze.
        Returns:
            dict: A dictionary with keys 'products' and 'prices', each mapping to a list of extracted values.
        """

        def detokenize(fragment_list):
            return "".join(t.replace("▁", "") for t in fragment_list).strip()

        result = predict_with_logits(self.model, self.tokenizer, self.id2label, text)
        tokens = result["tokens"]
        labels = result["predicted_labels_word_level"]
        word_ids = result["word_ids"]

        # 🪵 DEBUG PRINTS:
        #print("\n📨 Raw text:")
        #print(text)
        #print("\n🔤 Tokens:")
        #print(tokens)
        #print("\n🏷️ Labels:")
        #print(labels)

        entities = {
            "product": [],
            "price": [],
            "loc": [],
            "phone": []
        }

        current_entity = []
        current_tag = None
        previous_word_id = None
        reconstructed_word = ""

        for token, label, word_id in zip(tokens, labels, word_ids):
            if word_id is None:
                continue

            if word_id != previous_word_id:
                if current_entity and current_tag:
                    entities[current_tag].append(detokenize(current_entity))

                current_entity = [token.replace("▁", "")]
                current_tag = label[2:].lower() if label.startswith("B-") else None
            else:
                if label.startswith("I-") and current_tag:
                    current_entity.append(token.replace("▁", ""))
            previous_word_id = word_id

        if current_entity and current_tag:
            entities[current_tag].append(detokenize(current_entity))

        #🪵 DEBUG PRINTS:
        # 🪵 Final extracted entities:
        #print("\n📦 Extracted entities:")
        #print(entities)

        return entities

    def enrich_csv(self):
        """
        Processes the input CSV and writes a new CSV with extracted NER columns added.
        """
        self.output_path = os.path.join(os.path.dirname(os.getcwd()), 
                                        'data', 'enriched_telegram_data.csv')

        print("\nVerifying paths...")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"\nCSV not found at {self.csv_path}")
        
        try: 
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        except Exception as e:
            print(f"\nFailed to create output folder: {self.output_path}")
            raise e
        
        print("\nStarting enrichment process...")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as infile, \
                    open(self.output_path, 'w', encoding='utf-8', newline='') as outfile:

                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames + ['NER_Products', 'NER_Prices']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for i, row in enumerate(reader, 1):
                # DEBUG loop
                #last_rows = list(reader)[:1]  # Try last 10 rows
                #for i, row in enumerate(last_rows, 1):

                    text = row.get("Message", "")
                    ner_output = self.extract_entities(text)
                    row["NER_Products"] = ", ".join(ner_output.get("product", []))
                    row["NER_Prices"] = ", ".join(ner_output.get("price", []))
                    writer.writerow(row)

                    if i % 500 == 0:
                        print(f"\nProcessed {i} rows...")
        except Exception as e:
            print("\nAn error occurred during CSV processing.")
            raise e
        
        rel_ouput_path = os.path.relpath(self.output_path, os.getcwd())
        print(f"\nEnriched CSV saved to: {rel_ouput_path}")