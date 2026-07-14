from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Saare stored chunks dekho
for doc_id, doc in db.docstore._dict.items():
    print("ID:", doc_id)
    print("Chunk:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 80)