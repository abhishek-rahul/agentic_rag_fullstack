import pickle
from pathlib import Path

index_dir = Path(__file__).parent
pkl_path = index_dir / "index.pkl"

# Sirf apni trusted local file par pickle load karein
with open(pkl_path, "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

print("Total chunks:", len(index_to_docstore_id))

for index_position, doc_id in index_to_docstore_id.items():
    doc = docstore.search(doc_id)

    print(f"\n--- Chunk {index_position} ---")
    print("Document ID:", doc_id)
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)