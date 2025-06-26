import numpy as np
from lime.lime_text import LimeTextExplainer
from IPython.display import display, HTML


class LimeNERExplainer:
    def __init__(self, model, tokenizer, id2label, class_labels = None, max_tokens = 50):
        """
        Initialises the LIME explainer for NER token-level predictions.
        
        Args:
            model: Hugging Face model (e.g., AutoModelForTokenClassification).
            tokenizer: Corresponding tokenizer.
            id2label: Dictionary mapping label IDs to label strings.
            class_labels: Optional list of label strings for visualisation.
            max_tokens: Fixed token length for LIME input padding (default = 30).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_tokens = max_tokens
        self.class_labels = class_labels or sorted(set(id2label.values()))

        self.explainer = LimeTextExplainer(
            split_expression = r"\s+",
            bow = False,
            class_names = self.class_labels
        )

    def predict_proba(self, texts, predictor_fn):
        """
        Wraps your model to return per-token softmax probabilities for LIME.
        
        Args:
            texts: list of string inputs (space-separated tokens).
            predictor_fn: function like predict_with_logits.
        Returns:
            np.array of padded prediction confidences.
        """
        all_probs = []
        max_len = 0
        per_token_scores_list = []

        for text in texts:
            tokens = text.strip().split()
            output = predictor_fn(self.model, self.tokenizer, self.id2label, tokens)
            scores = [p.max() for p in output["probabilities"][1:-1]]  # ignore CLS/SEP
            per_token_scores_list.append(scores)
            max_len = max(max_len, len(scores))  # track longest

        for scores in per_token_scores_list:
            padded = np.pad(scores, (0, max_len - len(scores)), constant_values=0)
            all_probs.append(padded)
        return np.vstack(all_probs)


    def explain(self, text, predictor_fn, num_features = 6, num_samples = 100):
        """
        Runs the LIME explanation and returns an interactive visualisation.
        
        Args:
            text (str): Space-separated tokens.
            predictor_fn (func): e.g. predict_with_logits.
        """
        return self.explainer.explain_instance(
            text_instance = text,
            classifier_fn = lambda x: self.predict_proba(x, predictor_fn),
            num_features = num_features,
            num_samples = num_samples
        )
    
    def visualise_explanation(self, text, explanation):
        """
        Generates a color-coded HTML representation of token importances.

        Args:
            text (str): Original space-separated text.
            explanation: LIME explanation object from explain_instance().
        """
        token_weights = dict(explanation.as_list())
        tokens = text.strip().split()
        
        def color_token(token):
            weight = token_weights.get(token, 0)
            # Color: green for positive, red for negative, transparency based on magnitude
            color = "rgba(0, 200, 0, {:.2f})".format(min(abs(weight), 1.0)) if weight > 0 else \
                    "rgba(255, 0, 0, {:.2f})".format(min(abs(weight), 1.0)) if weight < 0 else \
                    "transparent"
            return f"<span style='background-color:{color}; padding:2px; margin:1px;'>{token}</span>"

        styled_tokens = [color_token(token) for token in tokens]
        html = "<div style='font-family:monospace; line-height:2em;'>{}</div>".format(" ".join(styled_tokens))
        display(HTML(html))

