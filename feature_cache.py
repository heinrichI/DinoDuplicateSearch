"""
Feature Cache - SQLite persistent cache for DINOv2 embeddings and SIFT features
"""
import os
import sqlite3
import struct
import zlib
from typing import Tuple, Optional, Set, List

import cv2
import numpy as np


def _keypoints_to_bytes(keypoints: Tuple[cv2.KeyPoint, ...]) -> bytes:
    """Serialize list of cv2.KeyPoint to bytes"""
    data = bytearray()
    # Write count as int32
    data.extend(struct.pack('<i', len(keypoints)))
    for kp in keypoints:
        data.extend(struct.pack(
            '<fffiiff',  # 3 floats + 2 ints + 2 floats
            kp.angle,
            kp.size,
            kp.response,
            kp.octave,      # int
            kp.class_id,    # int
            kp.pt[0],
            kp.pt[1]
        ))
    return bytes(data)


def _bytes_to_keypoints(data: bytes) -> Tuple[cv2.KeyPoint, ...]:
    """Deserialize bytes back to list of cv2.KeyPoint"""
    if not data:
        return ()
    offset = 0
    count = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    keypoints = []
    for _ in range(count):
        angle, size, response, octave, class_id, pt_x, pt_y = struct.unpack_from(
            '<fffiiff', data, offset
        )
        offset += struct.calcsize('<fffiiff')
        kp = cv2.KeyPoint(
            x=pt_x,
            y=pt_y,
            size=size,
            angle=angle,
            response=response,
            octave=int(octave),
            class_id=int(class_id)
        )
        keypoints.append(kp)
    return tuple(keypoints)


def _descriptors_to_bytes(descriptors: np.ndarray) -> bytes:
    """Serialize numpy array of descriptors to bytes"""
    return descriptors.tobytes()


def _bytes_to_descriptors(data: bytes, shape_str: str) -> np.ndarray:
    """Deserialize bytes back to numpy array"""
    shape = tuple(int(s) for s in shape_str.split(','))
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


class FeatureCache:
    """SQLite persistent cache for image features (embeddings + SIFT)"""

    COMPRESSION_LEVEL = 6

    def __init__(self, db_path: str = "feature_cache.db"):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database connection and create tables"""
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Needed for WAL mode + background thread
            timeout=30
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sift (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                keypoints BLOB,
                keypoints_count INTEGER DEFAULT 0,
                descriptors BLOB,
                descriptors_shape TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def get_embedding(self, path: str) -> Optional[Tuple[float, np.ndarray]]:
        """Get cached embedding for a path. Returns (mtime, embedding) or None."""
        cursor = self._conn.execute(
            "SELECT mtime, embedding FROM embeddings WHERE path = ?",
            (path,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        mtime, blob = row
        embedding = np.frombuffer(zlib.decompress(blob), dtype=np.float32).copy()
        return (mtime, embedding)

    def set_embedding(self, path: str, mtime: float, embedding: np.ndarray):
        """Save embedding to cache"""
        blob = zlib.compress(embedding.astype(np.float32).tobytes(), self.COMPRESSION_LEVEL)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (path, mtime, embedding) VALUES (?, ?, ?)",
            (path, mtime, blob)
        )
        self._conn.commit()

    def get_sift(self, path: str) -> Optional[Tuple[float, Tuple[cv2.KeyPoint, ...], np.ndarray]]:
        """Get cached SIFT features for a path. Returns (mtime, keypoints, descriptors) or None."""
        cursor = self._conn.execute(
            "SELECT mtime, keypoints, keypoints_count, descriptors, descriptors_shape FROM sift WHERE path = ?",
            (path,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        mtime, kp_blob, kp_count, des_blob, des_shape_str = row
        keypoints = _bytes_to_keypoints(zlib.decompress(kp_blob))
        descriptors = _bytes_to_descriptors(zlib.decompress(des_blob), des_shape_str) if des_blob else None
        return (mtime, keypoints, descriptors)

    def set_sift(
        self,
        path: str,
        mtime: float,
        keypoints: Tuple[cv2.KeyPoint, ...],
        descriptors: np.ndarray
    ):
        """Save SIFT features to cache"""
        kp_blob = zlib.compress(_keypoints_to_bytes(keypoints), self.COMPRESSION_LEVEL)
        des_blob = zlib.compress(_descriptors_to_bytes(descriptors), self.COMPRESSION_LEVEL) if descriptors is not None else None
        des_shape = f"{descriptors.shape[0]},{descriptors.shape[1]}" if descriptors is not None else "0,0"

        self._conn.execute(
            """INSERT OR REPLACE INTO sift
               (path, mtime, keypoints, keypoints_count, descriptors, descriptors_shape)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (path, mtime, kp_blob, len(keypoints), des_blob, des_shape)
        )
        self._conn.commit()

    def clear_all(self) -> int:
        """Delete all cached entries and VACUUM to reclaim disk space.
        
        Returns:
            Number of bytes freed (approximate, based on page count before/after).
        """
        before_pages = self._conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]

        self._conn.execute("DELETE FROM embeddings")
        self._conn.execute("DELETE FROM sift")
        self._conn.commit()
        self._conn.execute("VACUUM")
        self._conn.commit()

        after_pages = self._conn.execute("PRAGMA page_count").fetchone()[0]
        freed = (before_pages - after_pages) * page_size
        return freed

    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None