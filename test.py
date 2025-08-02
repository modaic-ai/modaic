from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

query = "Tell me about Apple's products"
results = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=[
"Apple is a popular fruit known for its sweetness and crisp texture.",	
"Apple is known for its innovative products like the iPhone.",
"Many people enjoy eating apples as a healthy snack.",
"Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
"An apple a day keeps the doctor away, as the saying goes.",
    ],
    top_n=3,
    return_documents=True,
    parameters= {
        "truncate": "END"
    }
)

print(query)
for r in results.data:
    print(r)
    print(r.score, r.document.text)