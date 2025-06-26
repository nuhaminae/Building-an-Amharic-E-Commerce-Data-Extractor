# Building an Amharic E-Commerce Data Extractor

This repository provides a comprehensive, reproducible framework for extracting, processing, and annotating e-commerce data in Amharic, with an emphasis on informal marketplaces such as Telegram channels. The pipeline is crafted to empower research and applications in natural language processing (NLP), information extraction, and digital commerce analytics for the Amharic language.

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

E-commerce in Ethiopia is rapidly evolving, with a significant amount of trading activity occurring via informal digital channels such as Telegram groups. Unfortunately, there is a dearth of structured datasets and language resources for Amharic, which impedes the development of robust commerce and NLP solutions.

This project tackles that gap by providing:

- A robust pipeline to **scrape, preprocess, and structure data** from Amharic Telegram channels.
- Tools and annotated corpora for **developing and evaluating Named Entity Recognition (NER) and information extraction models** in Amharic.
- Example scripts and Jupyter Notebooks for **exploratory data analysis and downstream NLP applications**.

---

## Features

- **Data Extraction:** Scripts for scraping raw messages (text, images, metadata) from Telegram channels, focusing on Amharic commerce content.
- **Preprocessing:** Advanced cleaning and structuring of raw chat data, including tokenisation, currency normalisation, and entity pattern extraction.
- **NER Annotation:** High-quality CoNLL-format annotation of Amharic e-commerce data for entities such as products, prices, phone numbers, and locations.
- **DVC Integration:** [Data Version Control (DVC)](https://dvc.org/) is used for tracking large, processed datasets and ensuring reproducibility.
- **Reproducible Workflows:** Jupyter Notebooks and modular Python scripts allow for easy experimentation and extension.
- **Continuous Integration:** Automated environment setup and basic validation via GitHub Actions.

---

## Directory Structure

```
.
├── data/
│   ├── raw/                          # Raw Telegram data (scraped messages, session files)
│   ├── processed_telegram_data.csv    # Cleaned, structured Telegram data (tracked with DVC)
│   ├── enriched_telegram_data.csv     # Further enriched dataset (for advanced analysis)
│   ├── ner_amharic_conll.txt          # NER-annotated Amharic text in CoNLL format
│
├── notebooks/                        # Jupyter Notebooks for data exploration, analysis, and modelling
├── scripts/                          # Python scripts for scraping, preprocessing, and annotation
├── models/                           # Model training outputs (for downstream tasks)
├── .github/workflows/                # CI/CD configuration files
├── .dvc/                             # DVC configuration and metadata
├── .gitignore                        # Git ignore rules
├── README.md                         # This documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or above
- [pip](https://pip.pypa.io/en/stable/)
- [DVC](https://dvc.org/) (for managing and retrieving large data files)
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

1. **Scrape and preprocess Telegram data:**
   - Use scripts in `scripts/` to extract and clean marketplace messages.
   - Raw data is saved in `data/raw/`, structured outputs in `data/`.

2. **Explore and annotate data:**
   - Use Jupyter Notebooks in `notebooks/` for exploratory analysis, visualisation, or model training.
   - The NER-labelled file `data/ner_amharic_conll.txt` enables NER experiments and model evaluation.

3. **NER Training Example:**
   The provided CoNLL file supports training entity recognition models. Example lines:
   ```
   የፀጉር    B-Product
   ማበጠርያ  I-Product
   ዋጋ      B-Price
   200      I-Price
   ብር      I-Price
   አዲስ      B-Loc
   አበባ    I-Loc
   ```

---

## Data

- **Raw Telegram data:** Not included in the repository for privacy. Instructions and scripts are provided for custom scraping.
- **Processed CSV:** Cleaned, structured data for analysis and machine learning.
- **NER Corpus:** Annotated Amharic e-commerce text for entity extraction research.

---

## NER Annotation Format

- Each token is on its own line, followed by its tag.
- Blank lines separate sentences/messages.
- Tag set includes:
  - `B-Product`/`I-Product`: Product entities
  - `B-Price`/`I-Price`: Price mentions
  - `B-Loc`/`I-Loc`: Location mentions
  - `B-Phone`: Phone numbers
  - `O`: Other/non-entity tokens

---

## Testing & CI

- GitHub Actions (`.github/workflows/CI.yml`) sets up the Python environment and runs basic validation on each push or pull request.
- Scripts and notebooks should be accompanied by relevant tests for robust development.

---

## Contribution

Contributions are very welcome! Please:
- Raise issues for bugs, suggestions, or feature requests.
- Fork the repository, create a branch, and submit a pull request.
- Adhere to best practices for code clarity, documentation, and testing.

---

## Project Status

The project is completed, checkout the full [commit history](https://github.com/nuhaminae/Building-an-Amharic-E-Commerce-Data-Extractor/commits?author=nuhaminae) here.


---

*Created and maintained by [nuhaminae](https://github.com/nuhaminae)*