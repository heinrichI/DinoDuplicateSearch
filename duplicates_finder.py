"""
Duplicates Finder - Core logic for finding duplicate images
Uses DINOv2 embeddings + Agglomerative Clustering + SIFT/WGC geometric verification
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoImageProcessor, AutoModel
from check_geometric_consistency import extract_sift_features, check_geometric_consistency


def _read_image_cv2(path: str) -> np.ndarray:
    """Read image using cv2 - supports Unicode paths on Windows"""
    # Check if path has non-ASCII characters
    try:
        path.encode('ascii')
        ascii_path = True
    except UnicodeEncodeError:
        ascii_path = False
    
    if ascii_path:
        # Try cv2.imread for ASCII paths
        img = cv2.imread(path)
        if img is not None:
            return img
    
    # Fallback to PIL for Unicode paths or if cv2 failed
    pil_img = Image.open(path)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


MODEL_PATH = r'f:\E\SourceDuplicateImages\DinoDuplicateSearch\models--facebook--dinov2-base'


class UnionFind:
    """Union-Find data structure for grouping WGC-verified pairs"""
    
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
    
    def find(self, x: str) -> str:
        """Find root with path compression"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: str, y: str):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    
    def get_groups(self) -> Dict[str, List[str]]:
        """Get all groups keyed by root"""
        groups: Dict[str, List[str]] = {}
        for item in self.parent:
            root = self.find(item)
            if root not in groups:
                groups[root] = []
            groups[root].append(item)
        return groups


@dataclass
class DuplicatePair:
    """A pair of similar images with their similarity score"""
    path1: str
    path2: str
    similarity: float
    geometric_verified: bool = False
    geometric_angle: float = 0.0
    geometric_angle_votes: int = 0
    geometric_scale: float = 0.0
    geometric_scale_votes: int = 0


@dataclass
class DuplicateGroup:
    """A group of duplicate images"""
    cluster_id: int
    pairs: List[DuplicatePair] = field(default_factory=list)
    
    @property
    def paths(self) -> List[str]:
        """Get all unique paths in this group"""
        all_paths = set()
        for pair in self.pairs:
            all_paths.add(pair.path1)
            all_paths.add(pair.path2)
        return list(all_paths)
    
    @property
    def is_geometric_verified(self) -> bool:
        """Check if any pair in the group is geometrically verified"""
        return any(p.geometric_verified for p in self.pairs)
    
    @property
    def avg_similarity(self) -> float:
        """Average similarity across all pairs"""
        if not self.pairs:
            return 0.0
        return sum(p.similarity for p in self.pairs) / len(self.pairs)


class DuplicatesFinder:
    """Finds duplicate images using DINOv2 embeddings and optional geometric verification"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self._sift_cache: Dict[str, Tuple] = {}
        self._progress_callback = None  # Store callback for use in embed_image
    
    def _load_model(self):
        """Lazy loading of DINOv2 model"""
        if self.model is None:
            if self._progress_callback:
                self._progress_callback(0, "Loading DINOv2 model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._progress_callback:
                self._progress_callback(10, "Loading model processor...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            if self._progress_callback:
                self._progress_callback(50, "Loading model weights...")
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            if self._progress_callback:
                self._progress_callback(100, "Model loaded")
    
    @torch.no_grad()
    def embed_image(self, path: str) -> np.ndarray:
        """Extract DINOv2 CLS embedding from an image"""
        self._load_model()
        
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        cls = torch.nn.functional.normalize(cls, dim=-1)
        
        return cls[0].cpu().numpy()
    
    @staticmethod
    def list_images(folder: str) -> List[str]:
        """List all image files in a folder"""
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts
        ]
    
    def find_duplicates(
        self, 
        folder_path: str, 
        distance_threshold: float = 0.45,
        enable_geometric_check: bool = False,
        wgc_threshold: float = 0.3,
        progress_callback=None
    ) -> List[DuplicateGroup]:
        """
        Find duplicate images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            distance_threshold: Distance threshold for Agglomerative Clustering (0.3-0.7)
                              Lower = stricter clustering, Higher = looser
            enable_geometric_check: Enable SIFT/WGC geometric verification
            wgc_threshold: Minimum ratio of votes for WGC (0.1-0.9), lower = easier to pass
            progress_callback: Optional callback(percent, message) for progress updates
        
        Returns:
            List of DuplicateGroup objects
        """
        # Store for later use
        self._wgc_threshold = wgc_threshold
        self._progress_callback = progress_callback  # Store for use in _load_model
        
        # Step 1: List images
        if progress_callback:
            progress_callback(0, "Listing images...")
        
        paths = self.list_images(folder_path)
        if not paths:
            return []
        
        # Step 2: Extract embeddings
        if progress_callback:
            progress_callback(10, f"Extracting embeddings for {len(paths)} images...")
        
        # Extract embeddings with progress
        embs = []
        for i, p in enumerate(paths):
            embs.append(self.embed_image(p))
            if progress_callback:
                pct = 10 + int((i / len(paths)) * 30)
                basename = os.path.basename(p)
                progress_callback(pct, f"Stage: Creating embeddings ({i+1}/{len(paths)})\nFile: {basename}")
        
        embs = np.stack(embs).astype("float32")
        path_to_idx = {p: i for i, p in enumerate(paths)}
        
        # Step 3: Agglomerative Clustering
        if progress_callback:
            progress_callback(40, "Clustering images...")
        
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = agg.fit_predict(embs)
        
        # Step 4: Group by cluster
        from collections import defaultdict
        clusters = defaultdict(list)
        for p, lab in zip(paths, labels):
            clusters[int(lab)].append(p)
        
        # Step 5: Create all candidate pairs and run WGC
        if progress_callback:
            progress_callback(60, "Creating duplicate pairs...")
        
        # Union-Find to group WGC-verified pairs
        uf = UnionFind()
        all_pairs: List[Tuple[DuplicatePair, str, str]] = []  # (pair, path1, path2)
        
        total_pairs = sum(len(items) * (len(items) - 1) // 2 for items in clusters.values())
        pair_count = 0
        
        for cluster_id, items in clusters.items():
            if len(items) <= 1:
                continue
            
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    idx1 = path_to_idx[items[i]]
                    idx2 = path_to_idx[items[j]]
                    sim = float(np.dot(embs[idx1], embs[idx2]))
                    
                    pair = DuplicatePair(
                        path1=items[i],
                        path2=items[j],
                        similarity=sim
                    )
                    
                    # Geometric verification
                    if enable_geometric_check:
                        geo_ok, angle, scale, angle_votes, scale_votes = self._verify_geometric(items[i], items[j])
                        pair.geometric_verified = geo_ok
                        pair.geometric_angle = angle
                        pair.geometric_angle_votes = angle_votes
                        pair.geometric_scale = scale
                        pair.geometric_scale_votes = scale_votes
                        
                        # Union in UF if WGC passes
                        if geo_ok:
                            uf.union(items[i], items[j])
                        
                        # Update progress with WGC info
                        if progress_callback and pair_count % 10 == 0:
                            base_pct = 60 + int((pair_count / total_pairs) * 20) if total_pairs > 0 else 60
                            basename1 = os.path.basename(items[i])
                            basename2 = os.path.basename(items[j])
                            status = "PASS" if geo_ok else "FAIL"
                            progress_callback(base_pct, f"Stage: WGC verification\nFile: {basename1} vs {basename2} [{status}]")
                    
                    all_pairs.append((pair, items[i], items[j]))
                    pair_count += 1
        
        # Step 6: Build groups based on Union-Find (WGC connectivity)
        if progress_callback:
            progress_callback(80, "Building duplicate groups...")
        
        groups: List[DuplicateGroup] = []
        group_id = 0
        
        print(f"[DEBUG] enable_geometric_check = {enable_geometric_check}")
        
        if enable_geometric_check:
            # Group by Union-Find roots (WGC connectivity)
            uf_groups = uf.get_groups()
            print(f"[DEBUG] uf_groups = {uf_groups}")
            print(f"[DEBUG] all_pairs count = {len(all_pairs)}")
            verified_count = sum(1 for p, _, _ in all_pairs if p.geometric_verified)
            print(f"[DEBUG] verified pairs = {verified_count}")
            
            for root, group_paths in uf_groups.items():
                if len(group_paths) < 2:
                    continue
                
                group = DuplicateGroup(cluster_id=group_id)
                for pair, p1, p2 in all_pairs:
                    if pair.geometric_verified and p1 in group_paths and p2 in group_paths:
                        # Only add pair if both paths are in this group
                        if p1 in group_paths:
                            group.pairs.append(pair)
                            break  # Don't add same pair multiple times
                
                # Rebuild pairs for this group
                group = DuplicateGroup(cluster_id=group_id)
                seen = set()
                for pair, p1, p2 in all_pairs:
                    if pair.geometric_verified:
                        if p1 in group_paths and p2 in group_paths:
                            # Create unique key for pair (sorted paths)
                            key = tuple(sorted([p1, p2]))
                            if key not in seen:
                                seen.add(key)
                                group.pairs.append(pair)
                
                if group.pairs:
                    groups.append(group)
                    group_id += 1
        else:
            # Without WGC: use embedding similarity only
            for cluster_id, items in clusters.items():
                if len(items) <= 1:
                    continue
                group = DuplicateGroup(cluster_id=cluster_id)
                for pair, _, _ in all_pairs:
                    if any(p in items for p in [pair.path1, pair.path2]):
                        if pair.path1 in items and pair.path2 in items:
                            group.pairs.append(pair)
                if group.pairs:
                    groups.append(group)
        
        if progress_callback:
            progress_callback(100, f"Found {len(groups)} duplicate groups")
        
        return groups
    
    def _verify_geometric(self, path1: str, path2: str) -> Tuple[bool, float, float, int, int]:
        """Verify geometric consistency between two images using SIFT/WGC"""
        # Load images (supports Unicode paths)
        img1 = _read_image_cv2(path1)
        img2 = _read_image_cv2(path2)
        
        if img1 is None or img2 is None:
            return False, 0.0, 0.0, 0, 0
        
        # Extract SIFT features
        kp1, des1 = self._get_sift_features(path1)
        kp2, des2 = self._get_sift_features(path2)
        
        # Check geometric consistency with stored threshold
        threshold_ratio = getattr(self, '_wgc_threshold', 0.3)
        return check_geometric_consistency(kp1, des1, kp2, des2, threshold_ratio)
    
    def _get_sift_features(self, path: str) -> Tuple:
        """Get cached SIFT features for an image"""
        if path not in self._sift_cache:
            img = _read_image_cv2(path)
            if img is None:
                return None, None
            kp, des = extract_sift_features(img)
            self._sift_cache[path] = (kp, des)
        return self._sift_cache[path]
