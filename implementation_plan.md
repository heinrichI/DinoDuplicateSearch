# Implementation Plan

[Overview]
Адаптировать работу progress dialog в app.py для корректного отображения во время поиска дубликатов. Текущая реализация использует устаревший API и блокирующий синхронный вызов, что не позволяет обновлять диалог.

[Types]
- ProgressCallback: Callable[[float, str], None] - функция обратного вызова для обновления прогресса (принимает процент 0-100 и сообщение)
- AsyncSearchResult: List[DuplicateGroup] - результаты поиска дубликатов

[Files]
- app.py: Изменить способ отображения dialog и сделать вызов асинхронным
- duplicates_finder.py: Добавить async версию метода find_duplicates() или использовать thread pool

[Functions]
- Modified: `_on_find_duplicates_click()` в app.py
  - Текущий код: синхронный вызов `self.finder.find_duplicates()` блокирует UI
  - Изменение: запустить в asyncio thread pool executor
  
- Modified: `did_mount()` в app.py  
  - Текущий код: добавляет dialog в overlay
  - Изменение: добавить в page.overlay перед показом

[Classes]
- SearchTab в app.py (класс уже существует, нужно изменить методы)

[Dependencies]
- Добавление: `concurrent.futures.ThreadPoolExecutor` (встроенный модуль Python)

[Testing]
- Запустить app.py и проверить что progress dialog появляется и обновляется во время поиска

[Implementation Order]
1. Изменить `_on_find_duplicates_click()` для использования thread pool executor
2. Обновить progress_callback для корректного обновления dialog
3. Обновить `did_mount()` для добавления dialog в overlay при инициализации
4. Протестировать приложение