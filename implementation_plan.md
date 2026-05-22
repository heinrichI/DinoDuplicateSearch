# Implementation Plan

[Overview]
Добавить lossless сжатие BLOB-ов в FeatureCache через zlib и кнопку ручной очистки кеша в UI.

Существующая база feature_cache.db весит ~102 MB после сканирования 50 фото. 99% объёма занимают SIFT-дескрипторы — массивы float32 (128-dim), сохраняемые через `tobytes()` без сжатия. zlib level 6 сжимает такие данные в ~2-3 раза без потерь. Одновременно добавляется кнопка "Clear Cache" на экране Search, которая удаляет все записи из обеих таблиц и выполняет VACUUM, освобождая место на диске. Автоматическая очистка не делается — только по запросу пользователя через UI.

[Types]
Один новый enum не требуется, все изменения — в существующих классах и методах.

Новый метод `clear_all()` в `FeatureCache`:
  - Удаляет все строки из `embeddings` и `sift`.
  - Выполняет `VACUUM` для возврата места ОС.
  - Возвращает размер freed_bytes.

[Files]
Три файла изменяются, один остаётся без изменений.

- `feature_cache.py` (изменяется)
  - Во все методы, читающие BLOB-ы (`get_embedding`, `get_sift`), добавляется `zlib.decompress()`.
  - Во все методы, пишущие BLOB-ы (`set_embedding`, `set_sift`), добавляется `zlib.compress()`.
  - `_init_db()`: добавить PRAGMA `auto_vacuum=INCREMENTAL` для новых баз.
  - Новый метод `clear_all()`: DELETE FROM обеих таблиц + VACUUM.
  - Удалить `remove_entries_not_in()` — не используется.
  - Вспомогательная константа `COMPRESSION_LEVEL = 6`.

- `duplicates_finder.py` (изменяется)
  - Удалить блок очистки stale cache (строки 380-385, вызов `self._cache.remove_entries_not_in(path_set)`).
  - Удалить саму переменную `path_set`.

- `app.py` (изменяется)
  - В `SearchScreen` добавить метод `on_clear_cache()`:
    - Показывает Popup подтверждения "Очистить кеш? Это удалит все сохранённые эмбеддинги и SIFT-дескрипторы.".
    - При подтверждении — запускает `self.finder._cache.clear_all()` в background thread с ProgressPopup.
    - После завершения — закрывает popup.
  - Импорт добавить не требуется (`.kv` вызывает метод напрямую).

- `app.kv` (изменяется)
  - На SearchScreen, после кнопки `find_button`, добавить кнопку `Clear Cache`:
    - text: "Clear Cache"
    - background_color: 0.8, 0.2, 0.2, 1.0 (красный)
    - on_release: `root.on_clear_cache()`

[Functions]
- `FeatureCache.__init__` — модифицируется: добавить `COMPRESSION_LEVEL = 6`.
- `FeatureCache._init_db` — модифицируется: добавить `PRAGMA auto_vacuum=INCREMENTAL`.
- `FeatureCache.get_embedding` — модифицируется: `np.frombuffer(zlib.decompress(blob))`.
- `FeatureCache.set_embedding` — модифицируется: `zlib.compress(blob, COMPRESSION_LEVEL)`.
- `FeatureCache.get_sift` — модифицируется: `zlib.decompress(kp_blob)`, `zlib.decompress(des_blob)`.
- `FeatureCache.set_sift` — модифицируется: `zlib.compress(kp_blob)`, `zlib.compress(des_blob)`.
- `FeatureCache.remove_entries_not_in` — удаляется (больше не вызывается нигде).
- `FeatureCache.clear_all()` — новый метод: DELETE + VACUUM.
- `DuplicatesFinder.find_duplicates` — модифицируется: удаляются строки 380-385 (stale cache cleanup).
- `SearchScreen.on_clear_cache` — новый метод в `app.py`.

[Classes]
- `FeatureCache` в `feature_cache.py`:
  - Модифицируется: в `_init_db` добавляется `auto_vacuum=INCREMENTAL`.
  - Удаляется: метод `remove_entries_not_in`.
  - Добавляется: метод `clear_all(self) -> int`.
- `DuplicatesFinder` в `duplicates_finder.py`:
  - Модифицируется `find_duplicates`: убирается блок очистки stale cache (строки 380-385).
- `SearchScreen` в `app.py`:
  - Добавляется метод `on_clear_cache`.
- Никакие классы не удаляются.

[Dependencies]
Никакие внешние пакеты не требуются — `zlib` входит в стандартную библиотеку Python.

[Testing]
Никакие существующие тесты не затрагиваются, так как тестов в репозитории нет.

Валидация:
1. Запустить приложение, нажать Find Duplicates — убедиться, что поиск работает.
2. Проверить размер базы до/после сжатия: `feature_cache.db` должен уменьшиться в ~2-3 раза.
3. Нажать Clear Cache — убедиться, что после подтверждения база очищается и VACUUM возвращает место.
4. Проверить, что повторный поиск после очистки работает корректно (кеш пересоздаётся, сжатие применяется).

[Implementation Order]
Один коммит со всеми изменениями в логическом порядке.

1. Изменить `feature_cache.py`: добавить zlib сжатие во все 4 метода get/set, заменить `remove_entries_not_in` на `clear_all`, добавить `auto_vacuum=INCREMENTAL` в `_init_db`.
2. Изменить `duplicates_finder.py`: удалить stale cache cleanup (строки 380-385).
3. Изменить `app.py`: добавить метод `on_clear_cache` в SearchScreen.
4. Изменить `app.kv`: добавить кнопку Clear Cache в SearchScreen.
5. Проверить, что приложение запускается и все функции работают.