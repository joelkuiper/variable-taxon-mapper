import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

MESH_XML = Path("desc2025.xml")

tree = ET.parse(MESH_XML)
root = tree.getroot()

records = []
tree_to_ui = {}  # tree number -> MeSH descriptor id

for dr in root.findall("DescriptorRecord"):
    mesh_id = dr.findtext("DescriptorUI")
    label_el = dr.find("DescriptorName/String")
    label = label_el.text if label_el is not None else None

    # take scope note from first concept if present
    definition = None
    concept_list = dr.find("ConceptList")
    if concept_list is not None:
        first_concept = concept_list.find("Concept")
        if first_concept is not None:
            definition = first_concept.findtext("ScopeNote")

    tree_numbers = []
    tnl = dr.find("TreeNumberList")
    if tnl is not None:
        for tn in tnl.findall("TreeNumber"):
            tn_text = tn.text.strip()
            tree_numbers.append(tn_text)
            tree_to_ui[tn_text] = mesh_id

    records.append(
        {
            "mesh_id": mesh_id,
            "label": label,
            "definition": definition,
            "tree_numbers": tree_numbers,
        }
    )

def derive_parent_mesh_id(tree_number: str):
    if "." not in tree_number:
        return None
    parent_tn = ".".join(tree_number.split(".")[:-1])
    return tree_to_ui.get(parent_tn)

rows = []
for rec in records:
    mesh_id = rec["mesh_id"]
    label = rec["label"]
    definition = rec["definition"]
    tree_numbers = rec["tree_numbers"]

    parent = None
    if tree_numbers:
        parent = derive_parent_mesh_id(tree_numbers[0])

    rows.append(
        {
            "mesh_id": mesh_id,
            "label": label,
            "parent": parent,
            "definition": definition,
        }
    )

df = pd.DataFrame(rows, columns=["mesh_id", "label", "parent", "definition"])

print(df.head())
print(len(df), "descriptors loaded")
