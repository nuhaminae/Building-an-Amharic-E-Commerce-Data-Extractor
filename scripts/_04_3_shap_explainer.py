import shap
import torch
import numpy as np
from IPython.display import display, HTML

class ShapNERExplainer:
    def __init__(self, model, tokenizer, id2label):
        """
        Initializes a SHAP explainer for token classification.

        Args:
            model: Fine-tuned Hugging Face model.
            tokenizer: Tokenizer used with the model.
            id2label: Mapping from label IDs to label strings.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.id2label = id2label

        # Define a function that SHAP can call on raw text
        def forward_fn(texts):
            outputs = []
            for text in texts:
                tokens = text.strip().split()
                encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, padding=True)
                with torch.no_grad():
                    logits = model(**encoding).logits.squeeze(0)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    word_ids = encoding.word_ids()
                    mask = [i for i in range(len(word_ids)) if word_ids[i] is not None]
                    token_probs = probs[mask].cpu().numpy()  # shape: (num_tokens, num_labels)

                # Aggregate: Take average max probability across tokens as a scalar
                max_probs = np.max(token_probs, axis=1)  # max per token
                confidence = np.mean(max_probs)  # mean over tokens
                outputs.append([confidence])  # SHAP expects 2D shape (batch, 1)

            return np.array(outputs)

        self.explainer = shap.Explainer(forward_fn, tokenizer)

    def explain(self, text):
        """
        Computes SHAP token attributions for the given text.

        Args:
            text (str): Space-separated token sequence (e.g., "ባልቻ ሆስፒታል 3000 ብር")

        Returns:
            shap.Explanation: SHAP object with visualizable token importance
        """
        return self.explainer([text])

    def visualise_shap_explanation(self, text, shap_values):
        tokens = text.strip().split()
        attributions = shap_values.values[0][:len(tokens)]  # skip padding if any

        def colorise(token, weight, label=None):
            base = "rgba({r},{g},0,{a})"
            r, g = (255, 0) if weight < 0 else (0, 200)
            a = min(abs(weight), 1.0)
            tooltip = f" title='{label}'" if label else ""
            return f"<span style='background-color:{base.format(r=r, g=g, a=a)}; padding:2px'{tooltip}>{token}</span>"

        # Get predicted labels from SHAP data or model
        labels = [None] * len(tokens)  # or use model predictions if available

        highlighted = [colorise(tok, attr.sum(), label)
                    for tok, attr, label in zip(tokens, attributions, labels)]
        display(HTML(" ".join(highlighted)))
