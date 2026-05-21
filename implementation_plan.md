# Implementation Plan: SQLite Cache for DINOv2 Embeddings and SIFT Features

[Overview]
Добавить SQLite кэш для DINOv2 эмбеддингов и SIFT характеристик (keypoints + descriptors) в `DuplicatesFinder`, чтобы избежать пересчета при повторных запусках и между разными директориями.

Сейчас эмбеддинги DINOv2 вычисляются каждый раз заново (строка 216 `duplicates_finder.py`), а SIFT кэшируется только in-memory (`_sift_cache`, строка 132) и сбрасывается при перезапуске. Нужен персистентный SQLite кэш, который хранит оба типа данных, проверяет `mtime` файла и автоматически инвалидирует устаревшие записи.

База данных будет одна (`feature_cache.db`) в корне проекта, с таблицами для embeddings и sift, плюс поддержка многопоточного доступа через WAL режим.

[Types]
Новый класс `FeatureCache` с методами для get/set эмбеддингов и SIFT-дескрипторов.

- `FeatureCache` — менеджер SQLite кэша
  - `db_path: str` — путь к файлу БД (по умолчанию `feature_cache.db` в CWD)
  - `__init__(db_path: str = "feature_cache.db")` — открывает/создает БД, включает WAL
  - `get_embedding(path: str) -> Optional[Tuple[float, np.ndarray]]` — возвращает `(mtime, embedding)` или None
  - `set_embedding(path: str, mtime: float, embedding: np.ndarray)` — сохраняет эмбеддинг
  - `remove_entries_not_in(paths: Set[str])` — удаляет записи для файлов, которых больше нет
  - `get_sift(path: str) -> Optional[Tuple[float, Tuple, np.ndarray]]` — возвращает `(mtime, keypoints, descriptors)` или None
  - `set_sift(path: str, mtime: float, keypoints: Tuple, descriptors: np.ndarray)` — сохраняет SIFT
  - `close()` — закрывает соединение

Внутренние структуры:
- `_keypoints_to_bytes(kp: Tuple[cv2.KeyPoint]) -> bytes` — сериализует KeyPoint список в бинарный формат
  - Формат: `[count(int32)] + для каждого KeyPoint: [angle(float32), size(float32), response(float32), octave(int32), class_id(int32), pt_x(float32), pt_y(float32)]`
- `_bytes_to_keypoints(data: bytes) -> Tuple[cv2.KeyPoint]` — десериализует обратно
- `_descriptors_to_bytes(des: np.ndarray) -> bytes` — `des.tobytes()`
- `_bytes_to_descriptors(data: bytes, shape_str: str) -> np.ndarray` — `np.frombuffer(data).reshape(shape)`

Схема SQLite:
```sql
CREATE TABLE IF NOT EXISTS embeddings (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sift (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    keypoints BLOB,
    keypoints_count INTEGER DEFAULT 0,
    descriptors BLOB,
    descriptors_shape TEXT,  -- "N,128"
    created_at TEXT DEFAULT (datetime('now'))
);

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

[Files]
Один новый файл; minimal changes to existing files.

- **Новый файл: `feature_cache.py`** — класс `FeatureCache` с SQLite операциями
- **Изменения в `duplicates_finder.py`**:
  - Добавить `from feature_cache import FeatureCache` в импорты
  - Добавить `self._cache = FeatureCache()` в `__init__`
  - Модифицировать `embed_image()` — проверять кэш перед вычислением, сохранять после
  - Модифицировать `_get_sift_features()` — проверять кэш перед SIFT, сохранять после
  - Добавить очистку кэша от stale файлов после поиска
- **Без изменений**: `app.py`, `app.kv`, `check_geometric_consistency.py`, `config.json`

[Functions]
Только новые функции в новом файле; две существующие функции модифицируются.

Новые функции в `feature_cache.py`:
- `FeatureCache.__init__(self, db_path: str = "feature_cache.db")` — инициализация БД
- `FeatureCache._init_db(self)` — создание таблиц и прагм
- `FeatureCache.get_embedding(self, path: str) -> Optional[Tuple[float, np.ndarray]]`
- `FeatureCache.set_embedding(self, path: str, mtime: float, embedding: np.ndarray)`
- `FeatureCache.get_sift(self, path: str) -> Optional[Tuple[float, Tuple, np.ndarray]]`
- `FeatureCache.set_sift(self, path: str, mtime: float, keypoints: Tuple, descriptors: np.ndarray)`
- `FeatureCache.remove_entries_not_in(self, paths: Set[str])` — чистка
- `FeatureCache.close(self)` — закрытие коннекта
- `FeatureCache._keypoints_to_bytes(keypoints) -> bytes` — сериализация
- `FeatureCache._bytes_to_keypoints(data: bytes) -> Tuple`
- `FeatureCache._descriptors_to_bytes(descriptors) -> bytes`
- `FeatureCache._bytes_to_descriptors(data: bytes, shape_str: str) -> np.ndarray`

Модифицированные функции в `duplicates_finder.py`:
- `DuplicatesFinder.__init__` — добавляет `self._cache = FeatureCache()`
- `DuplicatesFinder.embed_image` — проверяет `mtime` файла, если кэш совпадает — возвращает сохраненный эмбеддинг; иначе вычисляет и сохраняет
- `DuplicatesFinder._get_sift_features` — проверяет `mtime`, если кэш актуален — возвращает сохраненные keypoints/descriptors; иначе извлекает SIFT и сохраняет
- `DuplicatesFinder.find_duplicates` — после завершения вызывает `self._cache.remove_entries_not_in(current_paths)` для очистки

[Classes]
Один новый класс; один существующий класс модифицируется.

Новый класс:
- **`FeatureCache`** (новый файл `feature_cache.py`) — обертка над sqlite3
  - `__init__(db_path)`, `close()`
  - `get_embedding(path)`, `set_embedding(path, mtime, embedding)`
  - `get_sift(path)`, `set_sift(path, mtime, keypoints, descriptors)`
  - `remove_entries_not_in(paths)`
  - Приватные методы сериализации

Модифицированный класс:
- **`DuplicatesFinder`** (файл `duplicates_finder.py`):
  - Поле `self._cache: FeatureCache` в `__init__`
  - `embed_image()` теперь использует кэш (функция остается методом класса)
  - `_get_sift_features()` теперь использует кэш вместо только `_sift_cache`
  - `find_duplicates()` в конце вызывает чистку кэша

[Dependencies]
Одна новая стандартная зависимость — sqlite3 (встроена в Python).

- `sqlite3` — встроенный модуль Python, не требует установки
- Никаких новых внешних пакетов
- Изменений в `install.txt` и `README.md` не требуется

[Testing]
Проверить корректность кэша на реальных данных.

- Запустить поиск на папке с изображениями, убедиться что результаты совпадают до и после добавления кэша (сравнить группы)
- Запустить повторно — проверить что эмбеддинги и SIFT берутся из кэша (время выполнения должно быть значительно меньше)
- Добавить новый файл в папку — проверить что он будет обработан, а старые взяты из кэша
- Удалить файл из папки — проверить что кэш очищается (запись удаляется при вызове remove_entries_not_in)
- Проверить многопоточный доступ (WAL mode)

[Implementation Order]
Создать сериализацию KeyPoints, затем FeatureCache, затем интеграция в DuplicatesFinder.

1. Создать `feature_cache.py`:
   - Написать `_keypoints_to_bytes` и `_bytes_to_keypoints` (сериализация KeyPoint)
   - Написать `_descriptors_to_bytes` и `_bytes_to_descriptors`
   - Написать `__init__`, `_init_db`, `close`
   - Написать `get_embedding` / `set_embedding`
   - Написать `get_sift` / `set_sift`
   - Написать `remove_entries_not_in`
2. Модифицировать `duplicates_finder.py`:
   - Добавить `import` и `self._cache` в `__init__`
   - Модифицировать `embed_image()`: добавить проверку mtime + кэш
   - Модифицировать `_get_sift_features()`: заменить in-memory cache на FeatureCache
   - Добавить вызов чистки кэша в `find_duplicates()`
3. Проверить синтаксис (`python -m py_compile feature_cache.py`)
4. Протестировать end-to-end запуск через app