from IPython.display import display
import pandas as pd
import re
import unicodedata
import string
import os

class preprocess_and_clean():
    def __init__(self, df_path, output_folder):
        """
        Initialise the  class with the necessary parameters.

        Args:
            df_path (str): The path to the DataFrame.
            output_folder (str): The path to the output folder
        """

        self.df_path = df_path
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(), "data")
        self.df_raw = None # Initialise DataFrame attribute
        self.df = None # Initialise processed DataFrame attribute

        # Load DataFrame if path is provided
        if self.df_path:
            self.load_df()

    def load_df(self):
        """
        Loads the DataFrame from the specified path.
        Assumes the file is in CSV format.
        """

        #Calculate the relative path
        relative_df_path = os.path.relpath(self.df_path, os.getcwd())

        if self.df_path and os.path.exists(self.df_path):
            try:
                self.df_raw = pd.read_csv(self.df_path)
                print(f"DataFrame loaded successfully from {relative_df_path}")
                print("\nDataFrame head:")
                display (self.df_raw.head())
                print("\nDataFrame shape:")
                display(self.df_raw.shape)
            except Exception as e:
                print(f"Error loading DataFrame from {relative_df_path}: {e}")
                self.df_raw = None
        elif self.df_path:
            print(f"Error: File not found at {relative_df_path}")
            self.df_raw = None
        else:
            print("No DataFrame path provided during initialisation.")
            self.df_raw = None
        return self.df_raw
    
    def normalise_amharic(self, text):
        """
        Normalises and removes emojis from the loaded DataFrame
        """
            
        if not isinstance(text, str) or text.strip() == "":
            return ""

        # Normalise Unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Replace Amharic punctuation with space
        text = re.sub('[፡፣፤፥።፦]', ' ', text)

        # Replace Latin punctuation  with space
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Remove emojis and pictographs
        emoji_pattern = re.compile(
            "["                               # begin character class
            "\U0001F600-\U0001F64F"           # Emoticons
            "\U0001F300-\U0001F5FF"           # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"           # Transport & Map Symbols
            "\U0001F1E0-\U0001F1FF"           # Flags
            "\U0001F700-\U0001F77F"           # Alchemical Symbols
            "\U0001F780-\U0001F7FF"           # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"           # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"           # Supplemental Symbols & Pictographs
            "\U0001FA00-\U0001FA6F"           # Chess Symbols
            "\U0001FA70-\U0001FAFF"           # Symbols & Pictographs Extended-A
            "\u2600-\u26FF"                   # Miscellaneous Symbols
            "\u2700-\u27BF"                   # Dingbats
            "\U000024C2-\U0001F251"           # Enclosed Alphanumeric Supplement
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(' ', text) # Emojis are replaced with space

        # Keep only Ge'ez alphabets and latin numbers
        text = re.sub(r"[^ሀ-፼0-9\s]+", '', text) 

        # Remove extra whitespace after cleaning
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
    def collapse_phone_numbers(self, text):
        """
        Merges valid phone numbers split by spaces.
        Handles:
        - +251 or 251 numbers with or without space
        - Local 09/07 numbers
        Skips lone '251' if not part of a number
        """
        text = str(text) if pd.notnull(text) else ""
        text = re.sub(r'[+()\-\u200e]', '', text)
        tokens = text.split()
        merged = []
        buffer = []

        def flush():
            nonlocal buffer
            candidate = ''.join(buffer)
            if (len(candidate) == 10 and candidate.startswith(('09', '07'))) or \
            (len(candidate) == 12 and candidate.startswith('251')):
                merged.append(candidate)
            else:
                merged.extend(buffer)
            buffer = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.isdigit():
                lookahead = tokens[i + 1] if i + 1 < len(tokens) else ''
                if not buffer:
                    if token.startswith(('09', '07')):
                        buffer.append(token)
                    elif token == '251' and lookahead.isdigit():
                        buffer.append(token)
                    else:
                        merged.append(token)
                else:
                    buffer.append(token)
                    combined = ''.join(buffer)
                    if len(combined) in [10, 12]:
                        flush()
                    elif len(combined) > 12:
                        flush()
                        if token.startswith(('09', '07', '251')):
                            buffer.append(token)
                        else:
                            merged.append(token)
            else:
                if buffer:
                    flush()
                merged.append(token)
            i += 1

        if buffer:
            flush()

        return ' '.join(merged)

    def merge_prefix_particles(self, text):
        """
        Merges common Amharic prepositions and possessives (like ከ, በ, የ, ለ, ስለ)
        with the word that follows.
        """
        particles = {"ከ", "በ", "የ", "ለ", "ስለ"}
        tokens = text.split()
        merged = []
        i = 0
        while i < len(tokens):
            if tokens[i] in particles and i + 1 < len(tokens):
                merged.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return ' '.join(merged)

    
    def clean_df (self):
        """
        Clean the loaded DataFrame.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        
        if self.df_raw is not None:
            # Assign df_raw DataFrame to df
            self.df = self.df_raw.copy()

        if self.df is not None:

            # Handle phone numbers
            self.df['Message'] = self.df['Message'].apply(self.collapse_phone_numbers)
            print ("\nPhone number separated by space are handled.")

            # Merge prefixes like "ከ አዲስ" → "ከአዲስ"
            self.df['Message'] = self.df['Message'].apply(self.merge_prefix_particles)
            print ("\nPrefixes are handled.")
            
            # Drop empty indexes where 'Message' column is empty
            self.df['Message'] = self.df['Message'].apply(self.normalise_amharic)
            print("\nEmojis and non Geʽez characters in 'Message' column are handeled.")

            # Filters out any rows where the 'Message' column is empty or only contains whitespace.
            self.df = self.df[self.df['Message'].str.strip() != ""]
            # Remove rows where the entire 'Message' consists of digits only.
            self.df = self.df[~self.df['Message'].str.match(r'^\d+$')]

            self.df = self.df.dropna(subset="Message").reset_index(drop=True)
            print("\nEmpty rows where 'Message' column is empty and or has just numeric values are dropped.")

            
            # Handle columns with only whitespace
            cols_to_drop = []
            for col in self.df.columns:
                # Check if all non-null values in the column are just whitespace or empty after strip
                if self.df[col].dropna().apply(lambda x: str(x).strip() == '').all():
                    cols_to_drop.append(col)

            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                print(f"\nDropped columns containing only whitespace: {cols_to_drop}")

            print("\nDataFrame head:")
            display (self.df.head())

            # DataFrame shape
            print("\nDataFrame shape:")
            display(self.df.shape)

            print("\nDataFrame Info:")
            self.df.info()

            # Create output folder if it doesn't exist
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            # Define DataFrame name
            df_name = os.path.join(self.output_folder, "processed_telegram_data.csv")

            # Calculate the relative path
            relative_path = os.path.relpath(df_name, os.getcwd())

            # Save processed data to CSV
            self.df.to_csv(df_name, index=False)
            print(f"\nPrrocessed DataFrame Saved to: {relative_path}")
            
            return self.df
        
        return None