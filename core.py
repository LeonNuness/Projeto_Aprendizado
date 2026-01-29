import kagglehub
import pandas as pd
import os
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

path = kagglehub.dataset_download(
    "sumitm004/arxiv-scientific-research-papers-dataset"
)
print("Dataset em:", path)

files = os.listdir(path)
print(files)

df = pd.read_csv(os.path.join(path, files[0]))
df.head()

df = df[["title", "summary", "authors"]].dropna()

df["texto"] = df["title"] + ". " + df["summary"]

artigos = []

for _, row in df.iterrows():
    autores = row["authors"].split(",")
    for autor in autores:
        artigos.append({
            "professor": autor.strip(),
            "texto": row["texto"]
        })

def preprocess(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", texto)
    return texto.strip()

#Tirar dúvida
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

artigos = artigos[:5000]  # ajuste se quiser

texts = [preprocess(a["texto"]) for a in artigos]

embeddings_artigos = model.encode(
    texts,
    show_progress_bar=True
)

prof_to_vectors = defaultdict(list)

for artigo, emb in zip(artigos, embeddings_artigos):
    prof_to_vectors[artigo["professor"]].append(emb)

vetores_professores = {
    prof: np.mean(vectors, axis=0)
    for prof, vectors in prof_to_vectors.items()
}

descricao_projeto = """
Sistema de recomendação acadêmica baseado em
aprendizado não supervisionado e embeddings semânticos
para análise de produção científica
"""

v_proj = model.encode(preprocess(descricao_projeto))

scores = []

for prof, v_prof in vetores_professores.items():
    sim = cosine_similarity(
        [v_proj],
        [v_prof]
    )[0][0]
    scores.append((prof, sim))

ranking = sorted(scores, key=lambda x: x[1], reverse=True)

for prof, score in ranking[:10]:
    print(f"{prof}: {score:.4f}")
