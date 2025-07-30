from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
model.save("./models/paraphrase-MiniLM-L3-v2")
