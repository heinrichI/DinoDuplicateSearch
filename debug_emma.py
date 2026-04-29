"""
Debug script to understand why KMeans found Emma Stone pair but FAISS didn't
"""
import torch
import numpy as np
from PIL import Image
import os

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dinov2 import load_dinov2_model, extract_dinov2_embeddings
from duplicates_finder import DuplicatesFinder
from find_similar_images import find_similar_images_faiss

# Load model
print("Loading DINOv2 model...")
model = load_dinov2_model()
processor = None  # Will be loaded by the model

# Image paths
img1 = r"f:\E\SourceAntiDupl\TestCropHard\emma stone katharine mcphee kat dennings anna faris rumer willis the house bunny.jpg"
img2 = r"f:\E\SourceAntiDupl\TestCropHard\emma stone.jpg"

# Check files exist
print(f"Image 1 exists: {os.path.exists(img1)}")
print(f"Image 2 exists: {os.path.exists(img2)}")

# Extract embeddings using both methods
print("\n--- Extracting with DINOv2 (CLS token) ---")
try:
    emb1 = extract_dinov2_embeddings([img1], model)[0]
    emb2 = extract_dinov2_embeddings([img2], model)[0]
    
    # Compute cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    l2_dist = np.linalg.norm(emb1 - emb2)
    
    print(f"CLS Cosine similarity: {cos_sim:.4f}")
    print(f"CLS L2 distance: {l2_dist:.4f}")
    print(f"Similarity (1-L2): {1 - l2_dist:.4f}")
except Exception as e:
    print(f"Error: {e}")

# Now test with FAISS
print("\n--- Testing with FAISS ---")

# Get all images in folder
folder = r"f:\E\SourceAntiDupl\TestCropHard"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_paths = [
    os.path.join(folder, f) 
    for f in os.listdir(folder) 
    if f.lower().endswith(image_extensions)
]

print(f"Total images in folder: {len(image_paths)}")

# Find indices of our images
idx1 = None
idx2 = None
for i, p in enumerate(image_paths):
    if "emma stone katharine" in p.lower():
        idx1 = i
    if p.lower().endswith("emma stone.jpg") and "katharine" not in p.lower():
        idx2 = i

print(f"Image 1 index: {idx1}")
print(f"Image 2 index: {idx2}")

if idx1 is not None and idx2 is not None:
    print("\n--- FAISS KMeans Clustering ---")
    try:
        from sklearn.cluster import AgglomerativeClustering
        
        # Get embeddings for all images
        embeddings = extract_dinov2_embeddings(image_paths, model)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Run FAISS KMeans
        import faiss
        d = embeddings.shape[1]
        k = 15  # Same as in duplicates_finder
        seed = 42
        
        # Random sample for initialization
        np.random.seed(seed)
        random_idx = np.random.choice(len(embeddings), min(k, len(embeddings)), replace=False)
        centroids = embeddings[random_idx]
        
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, index_init=centroids, seed=seed)
        kmeans.train(embeddings)
        
        cluster_labels = kmeans.assign(embeddings)[0]
        
        # Find which cluster each image is in
        cluster1 = cluster_labels[idx1]
        cluster2 = cluster_labels[idx2]
        
        print(f"Image 1 (emma stone katharine...) cluster: {cluster1}")
        print(f"Image 2 (emma stone.jpg) cluster: {cluster2}")
        print(f"Same cluster: {cluster1 == cluster2}")
        
        # Get all images in the same cluster as image 2
        cluster_images = [i for i, c in enumerate(cluster_labels) if c == cluster2]
        print(f"\nAll images in cluster {cluster2}:")
        for ci in cluster_images:
            print(f"  {image_paths[ci]}")
            
    except Exception as e:
        print(f"FAISS error: {e}")
        import traceback
        traceback.print_exc()