
# 🔎 CPE Matcher

**CPE Matcher** is a Python script that automatically enriches a CSV list of software names and versions by finding their best-matching [CPE (Common Platform Enumeration)](https://nvd.nist.gov/products/cpe) identifiers using a combination of exact version parsing and semantic similarity via embeddings.

---

## 🚀 Features

- Parses the official [MITRE CPE Dictionary](https://nvd.nist.gov/products/cpe) (v2.3 XML format).
- Matches software and version entries from a CSV file to their best corresponding CPE entry.
- Uses `sentence-transformers` for semantic similarity between software titles and [PEP 440](https://www.python.org/dev/peps/pep-0440/) for versions comparisons. 

---

## 📂 Project Structure

```
.
├── main.py                     # Main script
├── model/                      # SentenceTransformer model directory
├── input/
│   └── your_input_file.csv     # Input CSV file (required)
├── output/
│   └── updated_cots_data.csv   # Output CSV file with CPE codes
├── official-cpe-dictionary_v2.3.xml  # Official CPE dictionary (required)
└── soft_versions_db.pkl        # Cached database of software/CPE versions
```

---

## 📄 Input Format

Place your CSV file inside the `input/` folder. The file must contain the following columns:

| Name            | Version        |
|-----------------|----------------|
| Software name   | Version string |

---

## 📤 Output

The script creates an output file: `output/updated_cots_data.csv`, adding the following columns:

| Name            | Version        | CPE Code                   | CPE Title                      |
|-----------------|----------------|----------------------------|--------------------------------|
| openssl         | 1.1.1k         | cpe:2.3:a:openssl:openssl:... | OpenSSL Project OpenSSL 1.1.1k |

---

## 🧠 How It Works

1. **CPE Dictionary Parsing**  
   Parses and stores all `(vendor, product, version)` entries from the XML into a local pickle database.

2. **Semantic Name Matching**  
   Uses sentence embeddings to find the top 3 closest software titles to each entry in the CSV.

3. **Version Matching**  
   - Tries to match versions using strict `packaging.version` logic.
   - Falls back to semantic similarity between version strings if parsing fails.

4. **Final Match Selection**  
   - Prioritizes exact matches with highest title similarity.
   - Falls back to best approximate matches.

---

## ⚙️ Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

You also need to download and place a compatible SentenceTransformer model in the `model/` folder (e.g., `all-MiniLM-L6-v2`).


## 🛠 Usage

```bash
python main.py
```

Make sure the following files/folders exist before running:
- `input/your_input_file.csv` – your CSV file
- `official-cpe-dictionary_v2.3.xml` – from [MITRE](https://nvd.nist.gov/products/cpe)
- `model/` – pretrained SentenceTransformer model folder

---

## 💾 Caching

To speed up future executions, the parsed CPE dictionary is saved as `soft_versions_db.pkl`. This allows skipping XML parsing and embedding recomputation.
