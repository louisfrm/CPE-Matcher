import os
import pandas as pd
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import pickle
import tqdm
import torch
import io

# Initialize the SentenceTransformer model
model = SentenceTransformer("./model")

# Define input and output directories
input_dir = "input"
output_dir = "output"

# Ensure that the output directory exists (create if necessary)
os.makedirs(output_dir, exist_ok=True)

# Get the list of files in the input directory and select the first one
input_files = os.listdir(input_dir)
if not input_files:
    raise Exception("No file found in the 'input' directory")
input_file = os.path.join(input_dir, input_files[0])
print(f"Selected input file: {input_file}")

# Read the input CSV file (assumes same columns: Name and Version)
data_df = pd.read_csv(input_file)

# Initialize columns for CPE code and title
data_df["CPE Code"] = None
data_df["CPE Title"] = None

# File to store/read the computed CPE embeddings
pickle_file = "cpe_embeddings.pkl"


# Define a custom unpickler to load data on CPU if needed
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# Load precomputed CPE embeddings if available, otherwise compute and save them
if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        if not torch.cuda.is_available():
            cpe_data, cpe_embeddings = CPU_Unpickler(f).load()
        else:
            cpe_data, cpe_embeddings = pickle.load(f)
else:
    # Parse the official CPE dictionary XML file and extract codes and titles
    tree = ET.parse("official-cpe-dictionary_v2.3.xml")
    root = tree.getroot()

    cpe_data = []
    cpe_titles = []

    # Define XML namespaces
    namespaces = {
        "cpe": "http://cpe.mitre.org/dictionary/2.0",
        "cpe-23": "http://scap.nist.gov/schema/cpe-extension/2.3",
    }

    # Iterate over each CPE item in the XML file
    for item in root.findall(".//cpe:cpe-item", namespaces):
        title = item.find("cpe:title", namespaces)
        cpe23_item = item.find("cpe-23:cpe23-item", namespaces)

        if title is not None and cpe23_item is not None:
            title_text = title.text
            cpe_code = cpe23_item.attrib["name"]
            cpe_titles.append(title_text)
            cpe_data.append(
                (title_text, cpe_code)
            )  # Save the title and corresponding CPE code

    # Calculate embeddings for all CPE titles at once
    print("Calculating embeddings...")
    cpe_embeddings = model.encode(cpe_titles, convert_to_tensor=True)

    # Save the computed embeddings to a pickle file for future use
    with open(pickle_file, "wb") as f:
        pickle.dump((cpe_data, cpe_embeddings), f)

# Iterate over each row in the input DataFrame to compute its embedding and find the best matching CPE
for index, row in tqdm.tqdm(
    data_df.iterrows(), total=data_df.shape[0], desc="Finding best CPE codes"
):
    cots_text = f"{row['Name']} {row['Version']}"
    cots_embedding = model.encode(cots_text, convert_to_tensor=True)

    # Compute cosine similarities with all CPE embeddings
    similarities = util.cos_sim(cots_embedding, cpe_embeddings)[0]
    best_match_idx = similarities.argmax().item()

    # Assign the most similar CPE code and title to the current row
    data_df.at[index, "CPE Code"] = cpe_data[best_match_idx][1]
    data_df.at[index, "CPE Title"] = cpe_data[best_match_idx][0]

# Save the updated DataFrame to a CSV file in the output directory
output_file = os.path.join(output_dir, "updated_cots_data.csv")
data_df.to_csv(output_file, index=False)
print(f"Result saved in: {output_file}")
