# Building an Amharic E-Commerce Data Extractor

This repository provides a robust pipeline for extracting, processing, and annotating e-commerce data in Amharic, with a particular focus on Telegram-based marketplaces. It is designed to facilitate downstream tasks such as Named Entity Recognition (NER), price extraction, product recognition, and location identification from Amharic-language posts and chats.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [NER Annotation Format](#ner-annotation-format)
- [Testing & CI](#testing--ci)
- [Contribution](#contribution)
- [Project Status](#project-status)

---

## Project Overview

E-commerce activity in Ethiopia is increasingly taking place via informal channels, especially Telegram groups and channels. This project addresses the lack of structured datasets and tools for Amharic e-commerce by providing:

- A pipeline to **scrape, preprocess, and structure data** from Amharic Telegram channels.
- Tools and annotated corpora for **developing and evaluating NER and information extraction models** in the Amharic language.
- Sample scripts and Jupyter Notebooks for **exploratory analysis and downstream NLP tasks**.

---

## Features

- **Data Extraction:** Scripts for extracting raw messages and metadata from Telegram channels.
- **Preprocessing:** Cleaning and structuring of raw chat data, with support for handling Amharic text.
- **Named Entity Recognition (NER) Annotation:** Amharic text annotated in CoNLL format for products, prices, phone numbers, and locations.
- **DVC Integration:** Uses [Data Version Control (DVC)](https://dvc.org/) for managing large processed datasets.
- **Reproducible Analysis:** Example notebooks for data exploration and NLP experiments.
- **Continuous Integration:** Automated tests and environment setup via GitHub Actions.

---

## Directory Structure

```
.
├── data/
│   ├── raw/                          # Raw Telegram data (scraped messages, photos, sessions)
│   ├── processed_telegram_data.csv    # Cleaned and processed Telegram data (DVC tracked)
│   ├── ner_amharic_conll.txt          # NER-annotated Amharic corpus in CoNLL format
│
├── notebooks/                        # Jupyter Notebooks for data exploration & analysis
├── scripts/                          # Python scripts for scraping, preprocessing, annotation
├── .github/workflows/                # CI configuration
├── .dvc/                             # DVC configuration and tracking
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or above
- [pip](https://pip.pypa.io/en/stable/)
- [DVC](https://dvc.org/) (for large data file management)
- [Git](https://git-scm.com/)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nuhaminae/Building-an-Amharic-E-Commerce-Data-Extractor.git
   cd Building-an-Amharic-E-Commerce-Data-Extractor
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .amhvenv
   source .amhvenv/bin/activate     # On Unix/Mac
   .amhvenv\Scripts\activate        # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull large data files with DVC (if needed):**
   ```bash
   dvc pull
   ```

---

## Usage

1. **Scrape and preprocess data:**
   - Use scripts in `scripts/` to extract Telegram messages and preprocess them.
   - Raw and processed data will be stored in `data/raw/` and `data/`.

2. **Run analysis and annotation:**
   - Explore the data or train models using the notebooks in the `notebooks/` folder.
   - Use `data/ner_amharic_conll.txt` for developing and evaluating Amharic NER models.

3. **NER Training Example:**
   - The provided CoNLL file includes tags for:
     - `B-Product`, `B-Price`, `I-Price`, `B-Phone`, `B-Loc`, `I-Loc`, etc.
   - Example snippet:
     ```
     ፀጉር    B-Product
     ዋጋ      B-Price
     200      I-Price
     ብር      I-Price
     አዲስ      B-Loc
     አበባ    I-Loc
     ```

---

## Data

- **Raw Telegram data:** Not included by default for privacy. Structure provided for custom scraping.
- **Processed CSV:** Cleaned, structured data for analysis and model training.
- **NER Corpus:** Annotated Amharic e-commerce text for NER and entity extraction research.

---

## NER Annotation Format

The `data/ner_amharic_conll.txt` file follows the CoNLL-style format:
- Each token is on a separate line, followed by its entity tag.
- Blank lines separate sentences.
- Tag set includes:  
  - `B-Product`/`I-Product`: Product names
  - `B-Price`/`I-Price`: Price mentions
  - `B-Loc`/`I-Loc`: Location names
  - `B-Phone`: Phone numbers
  - `O`: Other

---

## Testing & CI

- GitHub Actions (`.github/workflows/CI.yml`) ensures environment is set up and basic tests are run on every push or pull request.
- Custom scripts/notebooks should include their own test cases for robust development.

---

## Contribution

Contributions are warmly welcomed! Please:
- Raise an issue for bugs, feature requests, or questions.
- Fork the repository, create your branch, and submit a pull request.
- Ensure code is well-commented and tested.

---

## Project Status
The project is still underway, check the [commit history](https://github.com/nuhaminae/Building-an-Amharic-E-Commerce-Data-Extractor/commits?author=nuhaminae) for full commit history. 
