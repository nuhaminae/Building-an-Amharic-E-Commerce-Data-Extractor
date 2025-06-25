from IPython.display import display
import pandas as pd
import os
import random
import re

class label_data:
    # Class‐level constants for tuning and reuse

    PREFIXES = ("ከ", "በ", "የ", "ለ", "ስለ")
    PHONE_PATTERN = re.compile(r"^0[79]\d{8}$")
    INTL_PHONE_PATTERN = re.compile(r"^251\d{9}$")

    # Location‐NER helpers
    SUFFIX_LOCATORS = {
        "ፎቅ", "ሞል", "ህንፃ", "አካባቢ", "ጎን",
        "ላይ", "አጠገብ", "ታች", "ውስጥ", "ቀጥታ"
    }
    DIRECTIONAL_PHRASES = {
        ("ወረድ", "ብሎ"), ("ገባ", "ብሎ"),
        ("ወደ", "ግራ"), ("ወደ", "ቀኝ")
    }
    NUMERIC_MARKERS = {
        "ሱቅ", "ሱቅ ቁጥር", "ሱቅ ቁ", "ሱቅቁ",
        "ቁጥር", "ቁ", "ሱ ቁ", "ቤት ቁጥር",
        "ቤት ቁ", "ቢሮ ቁጥር", "ቢሮ ቁ"
    }
    ADDR_KEYWORDS = {"አድራሻ", "አድራሻችን"}
    PRODUCTS = {
        "መኪና", "ሸቶ", "ሱሪ", "ጫማ", "ጫማዎች",
        "ስልክ", "ልብስ", "ቦዲ", "ሸሚዝ", "ቀሚስ",
        "ቲሸርት", "ቲቪ", "ካልሲ", "ፀጉር", "ሰዓት", "ቡና", "ላፕቶፕ"
    }
    PRICE_ROOTS = {"ብር"}
    PRICE_MODIFIERS = {"ነፃ"}
    PRICE_LABELS = {"ዋጋ", "ዋጋው", "ዋጋችን"}

    def __init__(self, df_path, output_folder=None):
        """
        Initialise with a CSV path and optional output folder.
        """
        self.df_path = df_path
        self.output_folder = output_folder or os.path.join(os.getcwd(), "data")
        self.df = None

        # Known multi‐token location phrases
        self.known_locs = {
            "አዲስ አበባ", "መገናኛ", "4 ኪሎ", "6 ኪሎ", "5 ኪሎ",
            "ፒያሳ", "ሜክሲኮ", "ጀሞ 1", "ጀሞ 2", "ጀሞ 3",
            "ጊዮርጊስ", "መርካቶ", "ሚካኤል", "አይር ጤና",
            "ባልቻ ሆስፒታል", "ፈረንሳይ", "ቤላ", "ቦሌ", "የካ",
            "ጉለሌ", "ፊጋ", "አትላስ", "ካዛንቺስ", "ባቡር ጣቢያ",
            "ሽሮሜዳ", "ገነት ሆቴል", "ቄራ", "ቡልጋርያ", "መነሃርያ",
            "ከለላ ህንፃ", "በፀጋ ህንፃ", "ጀርመን", "ፖስታ ቤት",
            "ልደታ", "ላፍቶ", "ለሚ ኩራ", "ንፋስ ስልክ", "ቂርቆስ",
            "አዲስ ከተማ", "አራዳ", "ዘፍመሽ ግራንድ ሞል", "ኪኔሬት ታዎር",
            "ኬኬር", "ደምበል", "ላፍቶ ሞል", "ሳር ቤት", "አምባሳደር ሞል",
            "ቀበና", "ድሬዳዋ", "ሀረር", "ኦሮሚያ", "አዳማ", "አማራ",
            "ባህርዳር", "ትግራይ", "መቀሌ", "ሲዳማ", "አዋሳ", "ሶማሌ",
            "ጅጅጋ", "ሰመራ", "ጋምቤላ", "አሶሳ", "ክፍለሀገር", "አድራሻ"
        }

        if self.df_path:
            self.load_df()

    def load_df(self):
        """
        Load the CSV into a pandas DataFrame.
        """
        rel_path = os.path.relpath(self.df_path, os.getcwd())
        if os.path.exists(self.df_path):
            try:
                self.df = pd.read_csv(self.df_path)
                print(f"Loaded DataFrame from {rel_path}")
                display(self.df.head())
                display(self.df.shape)
            except Exception as e:
                print(f"Error loading DataFrame: {e}")
        else:
            print(f"File not found: {rel_path}")
        return self.df

    @staticmethod
    def strip_prefix(token):
        """
        Remove a leading grammatical prefix if present.
        """
        for prefix in label_data.PREFIXES:
            if isinstance(token, str) and  token.startswith(prefix):
                return token[len(prefix):]
        return token

    def label_token(self, token, prev_token=None):
        """
        BIO‐style tagging for single tokens: phones, prices, products.
        Uses prefix‐stripped roots for all checks.
        """
        token = token.strip()
        root = self.strip_prefix(token)
        prev_root = self.strip_prefix(prev_token.strip()) if prev_token else ""

        # Phone
        if self.PHONE_PATTERN.match(root) or self.INTL_PHONE_PATTERN.match(root):
            return "B-PHONE"

        # Price
        if root in self.PRICE_ROOTS:
            return "B-PRICE" if not prev_root.isdigit() else "I-PRICE"
        if root == "ብቻ" and prev_root in self.PRICE_ROOTS:
            return "I-PRICE"
        if root == "ነፃ":
            return "B-PRICE"
        if root.isdigit():
            forbidden = self.PRICE_ROOTS.union(self.PRICE_MODIFIERS)
            # ❗ Prevent nonsense like: ነፃ 100
            if prev_root in self.PRICE_MODIFIERS:
                return "O"
            return "B-PRICE" if prev_root not in forbidden else "I-PRICE"
        if root in self.PRICE_LABELS:
            return "B-PRICE"

        # Product
        if root in self.PRODUCTS:
            return "B-PRODUCT" if prev_root != "የ" else "I-PRODUCT"
        if root in {"እቃ", "ዕቃ", "ዕቃው", "እቃችን", "ዕቃችን", "ዕቃዎቻች"}:
            return "I-PRODUCT"

        return "O"

    def match_multiword_entity(self, tokens):
        """
        Multi‐token LOC tagging: exact phrases, stripped‐root single tokens,
        suffix locators, numeric markers, directional phrases.
        """
        # 0. Normalise and strip prefixes once
        tokens = [t.strip() for t in tokens]
        roots  = [self.strip_prefix(t) for t in tokens]

        labels = ['O'] * len(tokens)
        addr_found = False
        prior_loc_found = False

        # 1. Multi‐token exact match
        for length in range(5, 0, -1):
            for i in range(len(tokens) - length + 1):
                span = ' '.join(tokens[i:i+length])
                if span in self.known_locs:
                    labels[i] = "B-LOC" if not prior_loc_found else "I-LOC"
                    labels[i+1:i+length] = ["I-LOC"] * (length - 1)
                    prior_loc_found = True

        # 2. Single‐token or stripped‐root loc names
        for i, root in enumerate(roots):
            if root in self.ADDR_KEYWORDS:
                labels[i] = "B-LOC"
                addr_found = True
            elif root in self.known_locs and labels[i] == 'O':
                labels[i] = "I-LOC" if addr_found or prior_loc_found else "B-LOC"
                prior_loc_found = True

        # 3. Suffix‐based location triggers
        for i, token in enumerate(tokens):
            if token in self.SUFFIX_LOCATORS:
                labels[i] = "I-LOC"

                if i > 0:
                    prev = self.strip_prefix(tokens[i - 1])
                    # Prevent location tagging if previous token is a phone number
                    if self.PHONE_PATTERN.match(prev) or self.INTL_PHONE_PATTERN.match(prev):
                        continue

                    if labels[i - 1] == "O":
                        labels[i - 1] = "I-LOC" if addr_found or prior_loc_found else "B-LOC"
                prior_loc_found = True


        # 4. Numeric markers (using stripped roots)
        for i, root in enumerate(roots):
            if root in self.NUMERIC_MARKERS:
                labels[i] = "I-LOC"
                if i+1 < len(roots) and roots[i+1].isdigit():
                    labels[i+1] = "I-LOC"
                if i>0 and roots[i-1].isdigit():
                    labels[i-1] = "I-LOC"
            if root in {"ተኛ", "ኛ"} and i>0 and roots[i-1].isdigit():
                labels[i] = labels[i-1] = "I-LOC"

        # 5. Directional two‐token phrases (using stripped roots)
        for i in range(len(tokens)-1):
            if (roots[i], roots[i+1]) in self.DIRECTIONAL_PHRASES and \
                labels[i] == labels[i+1] == 'O':
                labels[i] = "B-LOC"
                labels[i+1] = "I-LOC"

        return labels


    def generate_conll_sample(self, sample_size=40):
        """
        Sample messages, label them, and write CoNLL‐style output.
        """
        if self.df is None or 'Message' not in self.df.columns:
            print("DataFrame not available or missing 'Message' column.")
            return

        messages = self.df['Message'].dropna().unique().tolist()
        sample   = random.sample(messages, min(sample_size, len(messages)))

        os.makedirs(self.output_folder, exist_ok=True)
        out_path = os.path.join(self.output_folder, "ner_amharic_conll.txt")
        rel_path = os.path.relpath(out_path, os.getcwd())

        with open(out_path, 'w', encoding='utf-8') as f:
            for msg in sample:
                tokens = msg.strip().split()
                labels = self.match_multiword_entity(tokens)

                for i, token in enumerate(tokens):
                    prev  = tokens[i-1] if i>0 else None
                    label = labels[i] if labels[i] != 'O' else self.label_token(token, prev)
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
        
        print(f"\nHeuristically labeled CoNLL saved to: {rel_path}")
