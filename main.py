import os
import pandas as pd
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import pickle
import tqdm
from packaging.version import Version, InvalidVersion
import torch

# ==============================================================================
# UTILS
# ==============================================================================


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import io

            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def safe_pickle_load(file_path):
    with open(file_path, "rb") as f:
        if torch.cuda.is_available():
            return pickle.load(f)
        else:
            return CPU_Unpickler(f).load()


def safe_str(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Initialize the SentenceTransformer model
model = SentenceTransformer("./model")

# Define input and output directories
input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Get the list of files in the input directory and select the first one
input_files = os.listdir(input_dir)
if not input_files:
    raise Exception("No file found in the 'input' directory")
input_file = os.path.join(input_dir, input_files[0])
print(f"Selected input file: {input_file}")

# Read the input CSV file (assumes same columns: Name and Version)
data_df = pd.read_csv(input_file)
data_df["Name"] = data_df["Name"].astype(str).str.strip()
data_df["Version"] = data_df["Version"].apply(safe_str)
# Initialize result columns (to store the matching version and the CPE code)
data_df["CPE Code"] = None
data_df["CPE Title"] = None

# ==============================================================================
# HELPER FUNCTIONS FOR PARSING AND VERSION MATCHING
# ==============================================================================


def parse_version_str(version_str):
    """
    Extracts a PEP 440 Version object from a version string using packaging.version.

    Returns None if the string does not comply with PEP 440.
    """
    if not isinstance(version_str, str) or not version_str.strip():
        return None
    try:
        return Version(version_str.strip())
    except InvalidVersion:
        return None


def match_version(input_version, candidate_versions):
    """
    Compares the input version with a list of candidate versions (from the XML).

    Each candidate is a tuple: (version_str, parsed_version, cpe_code, full_title).

    If an exact match exists (i.e. same parsed version), it is returned.
    Otherwise, the candidate with the highest version not exceeding the input version is selected.
    If all candidate versions exceed the input, the smallest candidate is returned.

    This comparison relies on the built-in comparison features of packaging.version.
    If the input version can't be parsed or no candidate can be parsed,
    the comparison falls back to version string similarity using embeddings.
    """
    input_parsed = parse_version_str(input_version)
    # Separate candidates with a successfully parsed version
    standard_candidates = [cand for cand in candidate_versions if cand[1] is not None]

    if input_parsed is not None and standard_candidates:
        # Sort candidates in ascending order (thanks to built-in comparison)
        sorted_candidates = sorted(standard_candidates, key=lambda cand: cand[1])

        # If a candidate has the exact same version, return it
        for cand in sorted_candidates:
            if cand[1] == input_parsed:
                return cand

        # Otherwise, choose the candidate with the highest version not exceeding the input
        lower_candidates = [
            cand for cand in sorted_candidates if cand[1] < input_parsed
        ]
        if lower_candidates:
            return lower_candidates[-1]  # Last candidate before the input

        # If no candidate is below the input, return the smallest available one
        return sorted_candidates[0]

    # Fallback: use embedding model to compare version string similarity
    input_version_emb = model.encode(input_version, convert_to_tensor=True)
    best_sim = -1
    best_candidate = None
    for cand in candidate_versions:
        candidate_str = cand[0]
        candidate_emb = model.encode(candidate_str, convert_to_tensor=True)
        sim = util.cos_sim(input_version_emb, candidate_emb).item()
        if sim > best_sim:
            best_sim = sim
            best_candidate = cand
    return best_candidate


# ==============================================================================
# BUILDING THE SOFTWARE AND VERSION DATABASE (soft_versions_db)
# ==============================================================================
# A pickle file is used to avoid reparsing the XML on each run.
soft_db_pickle_file = "soft_versions_db.pkl"

if os.path.exists(soft_db_pickle_file):
    soft_versions_db = safe_pickle_load(soft_db_pickle_file)
else:
    soft_versions_db = {}
    # Parse the official CPE dictionary XML file
    tree = ET.parse("official-cpe-dictionary_v2.3.xml")
    root = tree.getroot()
    namespaces = {
        "cpe": "http://cpe.mitre.org/dictionary/2.0",
        "cpe-23": "http://scap.nist.gov/schema/cpe-extension/2.3",
    }

    # For each cpe-item in the XML, extract the code, title, and version
    for item in root.findall(".//cpe:cpe-item", namespaces):
        title_elem = item.find("cpe:title", namespaces)
        cpe23_item = item.find("cpe-23:cpe23-item", namespaces)
        if title_elem is not None and cpe23_item is not None:
            full_title = title_elem.text.strip()
            cpe_code = cpe23_item.attrib["name"]
            tokens = cpe_code.split(":")
            if len(tokens) < 6:
                continue  # unexpected format
            # For cpe:2.3, index 3 is vendor, 4 is product, 5 is version
            vendor = tokens[3]
            product = tokens[4]
            version_raw = tokens[5]
            # Clean the title by removing the version fragment (if present)
            clean_title = full_title.replace(version_raw, "").strip()
            # Parse the version into standard Version object (possibly None)
            parsed_version = parse_version_str(version_raw)
            # Candidate entry for a given version (can also store full cpe_code)
            candidate_entry = (version_raw, parsed_version, cpe_code, full_title)
            # Grouping key is the (vendor, product) pair
            key = (vendor, product)
            if key not in soft_versions_db:
                soft_versions_db[key] = {"clean_title": clean_title, "versions": []}
            soft_versions_db[key]["versions"].append(candidate_entry)

    # Compute the embedding of clean_title for each software (for name matching)
    for key in soft_versions_db:
        soft_versions_db[key]["embedding"] = model.encode(
            soft_versions_db[key]["clean_title"], convert_to_tensor=True
        )

    # Save the database for reuse
    with open(soft_db_pickle_file, "wb") as f:
        pickle.dump(soft_versions_db, f)

# ==============================================================================
# MATCHING PROCESS FOR EACH ROW IN THE CSV
# ==============================================================================
# Matching CPE codes for each software in the input CSV
for index, row in tqdm.tqdm(
    data_df.iterrows(), total=data_df.shape[0], desc="Matching CPE codes"
):
    input_name = row["Name"]
    input_version = row["Version"]
    print(f"\n\nInput: {input_name} - Version: {input_version}")

    # Compute the embedding of the input name
    input_name_emb = model.encode(input_name, convert_to_tensor=True)

    # Compute cosine similarity between input name and all known software titles
    sim_list = []
    for key, info in soft_versions_db.items():
        sim = util.cos_sim(input_name_emb, info["embedding"]).item()
        sim_list.append((key, sim))

    # Select the top 3 most similar software titles
    top3_keys = sorted(sim_list, key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 candidate titles:")
    for k, s in top3_keys:
        print(f"  - {soft_versions_db[k]['clean_title']} (score={s:.4f})")

    version_matches = []

    def version_distance(v1, v2):
        def version_to_tuple(v):
            return tuple(int(part) for part in str(v).split(".") if part.isdigit())

        t1 = version_to_tuple(v1)
        t2 = version_to_tuple(v2)
        length = max(len(t1), len(t2))
        t1 += (0,) * (length - len(t1))
        t2 += (0,) * (length - len(t2))
        return sum(abs(a - b) for a, b in zip(t1, t2))

    for key, name_sim in top3_keys:
        info = soft_versions_db[key]
        best_version = match_version(input_version, info["versions"])
        if best_version is not None:
            parsed_input_version = parse_version_str(input_version)

            if parsed_input_version is not None and best_version[1] is not None:
                if parsed_input_version == best_version[1]:
                    dist = 0
                else:
                    dist = version_distance(parsed_input_version, best_version[1])
            else:
                input_version_emb = model.encode(input_version, convert_to_tensor=True)
                best_version_emb = model.encode(best_version[0], convert_to_tensor=True)
                dist = 1 - util.cos_sim(input_version_emb, best_version_emb).item()

            version_matches.append((key, best_version, dist, name_sim))
            print(
                f"  → Version candidate from '{soft_versions_db[key]['clean_title']}': {best_version[0]} (CPE: {best_version[2]}) | dist={dist:.4f} | sim={name_sim:.4f}"
            )

    if version_matches:
        version_matches.sort(key=lambda x: x[2])
        min_dist = version_matches[0][2]

        # Check for multiple candidates with the same minimal distance
        tied = [vm for vm in version_matches if abs(vm[2] - min_dist) < 1e-6]

        if len(tied) == 1:
            selected = tied[0]
        else:
            # Break ties using the highest title similarity
            selected = max(tied, key=lambda x: x[3])
            print("  ⚠️ Tie detected, selecting candidate with best title similarity")

        best_key, best_version_entry, _, _ = selected
        data_df.at[index, "CPE Code"] = best_version_entry[2]
        data_df.at[index, "CPE Title"] = best_version_entry[3]
        print(
            f"✅ Final Match: {soft_versions_db[best_key]['clean_title']} {best_version_entry[0]} → {best_version_entry[2]}"
        )
    else:
        # No valid match found
        data_df.at[index, "CPE Code"] = None
        data_df.at[index, "CPE Title"] = None
        print("❌ No match found")


# ==============================================================================
# SAVE THE UPDATED DATAFRAME
# ==============================================================================
output_file = os.path.join(output_dir, "updated_cots_data.csv")
data_df.to_csv(output_file, index=False)
print(f"Result saved in: {output_file}")
