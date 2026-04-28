import os
import sys
# sys.path.append(r'C:\Python311\Lib\site-packages')
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# Создайте окружение:
# conda create -n faiss_env python=3.10 --solver=classic
# conda activate faiss_env
# Установите faiss-gpu:
# conda install -c conda-forge faiss-gpu --solver=classic
# Примечание: Conda автоматически подтянет необходимые библиотеки CUDA, совместимые с вашей системой.
# Проверка установки:
# python
# import faiss
# print(faiss.get_num_gpus()) # Должно вернуть число > 0

MODEL_ID = "facebook/dinov2-base"  # small/base/large models also exist on HF
device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir='cache')
# model = AutoModel.from_pretrained(MODEL_ID, cache_dir='cache').to(device)
processor = AutoImageProcessor.from_pretrained('f:\E\SourceDuplicateImages\DinoDuplicateSearch\models--facebook--dinov2-base')
model = AutoModel.from_pretrained('f:\E\SourceDuplicateImages\DinoDuplicateSearch\models--facebook--dinov2-base').to(device)
model.eval()

@torch.no_grad()
def embed_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    outputs = model(**inputs)
    # last_hidden_state: [batch, tokens, dim]
    feats = outputs.last_hidden_state

    cls = feats[:, 0]  # [batch, dim] - global embedding (CLS token) :contentReference[oaicite:6]{index=6}
    cls = torch.nn.functional.normalize(cls, dim=-1)  # good for cosine similarity
    return cls[0].cpu().numpy()

# vec = embed_image("my_image.jpg")
# print(vec.shape)

def list_images(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]

# 1) build embeddings
paths = list_images("f:\E\SourceAntiDupl\TestCropHard")
embs = np.stack([embed_image(p) for p in paths]).astype("float32")
# embs are already L2-normalized, so inner product == cosine similarity

# 2) build FAISS index (inner product)
# d = embs.shape[1]
# index = faiss.IndexFlatIP(d)
# index.add(embs)

# 3) query
# query_path = "query.jpg"
# q = embed_image(query_path).astype("float32")[None, :]
# scores, idxs = index.search(q, k=5)

# print("Query:", query_path)
# for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), start=1):
#     print(f"{rank:02d}. score={s:.3f}  {paths[i]}")


# Clustering images
from sklearn.cluster import KMeans
from collections import defaultdict

k = 10
kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
labels = kmeans.fit_predict(embs)

clusters = defaultdict(list)
for p, lab in zip(paths, labels):
    clusters[int(lab)].append(p)

for lab in range(k):
    print(f"\nCluster {lab} ({len(clusters[lab])} images)")
    for p in clusters[lab][:5]:
        print("  ", p)

# DEBUG: Show pairwise similarities within KMeans clusters
def print_cluster_pairwise_similarities(clusters, embs, paths, threshold):
    """Print FAISS similarity scores for all pairs within each KMeans cluster."""
    # Create path to index mapping
    path_to_idx = {p: i for i, p in enumerate(paths)}
    
    print("\n" + "="*60)
    print("KMEANS CLUSTER SIMILARITY DEBUG")
    print("="*60)
    
    for lab, items in clusters.items():
        if len(items) <= 1:
            continue
        print(f"\n--- Cluster {lab} DEBUG ({len(items)} images) ---")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx1, idx2 = path_to_idx[items[i]], path_to_idx[items[j]]
                sim = float(np.dot(embs[idx1], embs[idx2]))  # inner product = cosine for normalized
                status = "ABOVE" if sim > threshold else "BELOW"
                print(f"  [{sim:.4f}] {status} threshold")
                print(f"    A: {os.path.basename(items[i])}")
                print(f"    B: {os.path.basename(items[j])}")

# Build FAISS index for similarity computations
import faiss
d = embs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embs)

threshold_sim = 0.90

# Call debug function for KMeans clusters
print_cluster_pairwise_similarities(clusters, embs, paths, threshold_sim)

limit = 30  # макс. кол-во соседей на точку (с запасом)

# Поиск ближайших соседей (включая самого себя)
sims, idxs = index.search(embs, limit)

# Строим граф: ребро, если сходство > threshold и не петля
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
rows = []
cols = []
for i in range(len(embs)):
    for j, s in zip(idxs[i], sims[i]):
        if i == j: continue
        if s > threshold_sim:
            rows.append(i)
            cols.append(j)

adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(embs), len(embs)))
n_comp, labels = connected_components(csgraph=adj, directed=False)

from collections import defaultdict

# 1. Group file paths by their component label
clusters = defaultdict(list)
for i, label in enumerate(labels):
    clusters[label].append(paths[i])

# 2. Print only clusters that contain more than one image (actual duplicates)
print(f"\nFound {n_comp} total groups. Printing duplicate clusters:\n")

duplicate_count = 0
for label, items in clusters.items():
    if len(items) > 1:
        duplicate_count += 1
        print(f"--- Cluster {duplicate_count} ({len(items)} images) ---")
        for p in items:
            print(f"  {p}")
        print()

if duplicate_count == 0:
    print("No duplicates found with the current threshold.")