# Implementation Plan

[Overview]
Rewrite the Flet-based GUI (`app.py`) to use the Kivy framework, keeping `duplicates_finder.py` and all core logic unchanged.

The current application uses [Flet](https://flet.dev) (a Flutter-based Python UI framework). The task is to replace the entire GUI layer (`app.py`, `dialog.py`) with [Kivy](https://kivy.org), a mature cross-platform Python UI framework. The core logic in `duplicates_finder.py`, `check_geometric_consistency.py`, `main_console.py`, and `config.json` handling will remain untouched.

Kivy uses a different paradigm than Flet:
- Layout is declarative via `.kv` language files or inline `Builder.load_string()`
- Widgets inherit from `kivy.uix.*` classes
- Async work is done via `threading.Thread` + `Clock.schedule_once` for UI updates (Kivy is not asyncio-native)
- No concept of `page.update()` — widgets update reactively when properties change
- File chooser is `FileChooserIconView` / `FileChooserListView` inside a `Popup`
- Progress dialogs use `Popup` with a `ProgressBar`
- Images displayed via `kivy.uix.image.AsyncImage` or `Image`

[Types]
No new types are introduced; existing `DuplicateGroup` and `DuplicatePair` dataclasses from `duplicates_finder.py` are reused as-is.

All data types remain in `duplicates_finder.py`:
- `DuplicatePair` — path1, path2, similarity, geometric fields
- `DuplicateGroup` — cluster_id, pairs list, computed properties
- `UnionFind` — internal grouping structure

[Files]
Replace `app.py` and `dialog.py` with a new Kivy-based `app.py`; add a `app.kv` layout file.

- **`app.py`** — completely rewritten using Kivy; old Flet code removed
- **`app.kv`** — new file; Kivy language layout definitions for all screens/widgets
- **`dialog.py`** — replaced with a simple Kivy demo (or deleted; it was not used by main app)
- **`duplicates_finder.py`** — no changes
- **`check_geometric_consistency.py`** — no changes
- **`config.json`** — no changes (same load/save logic)
- **`install.txt`** — updated: replace `flet` with `kivy` install instructions
- **`README.md`** — updated to reflect Kivy usage

[Functions]
Replace all Flet event handlers and UI builder methods with Kivy equivalents.

New / replaced functions in `app.py`:
- `load_config() -> dict` — unchanged logic, kept as-is
- `save_config(config: dict)` — unchanged logic, kept as-is
- `open_original_image(path: str)` — unchanged, still uses `os.startfile`
- `DinoDuplicateApp.build()` — Kivy App entry point; returns root widget loaded from `app.kv`
- `SearchScreen.on_browse_click()` — opens `FileChooserPopup`, replaces `_on_browse_click`
- `SearchScreen.on_find_duplicates()` — starts background thread, shows progress popup; replaces `_on_find_duplicates_click`
- `SearchScreen._run_search_thread(folder, threshold, geo_check, wgc_threshold)` — background thread function; calls `finder.find_duplicates()` with a progress callback
- `SearchScreen._update_progress(percent, message)` — called via `Clock.schedule_once` from thread; updates `ProgressPopup`
- `SearchScreen._on_search_complete(results)` — called via `Clock.schedule_once`; dismisses popup, switches to results screen
- `ResultsScreen.update_results(groups)` — populates `ScrollView` with group cards; replaces `ResultsTab.update_results`
- `ResultsScreen._build_group_card(group, index)` — builds a `BoxLayout` card for each group; replaces `_create_group_card`
- `FileChooserPopup.__init__()` — `Popup` subclass with `FileChooserIconView`; replaces Flet `FilePicker`
- `ProgressPopup.__init__()` — `Popup` subclass with `ProgressBar` and status labels

[Classes]
Replace Flet Tab/Container classes with Kivy Screen and Widget classes.

New classes in `app.py`:
- **`DinoDuplicateApp(App)`** — main Kivy application; `build()` returns `ScreenManager`
- **`SearchScreen(Screen)`** — replaces `SearchTab`; contains directory input, sliders, switches, browse/find buttons
- **`ResultsScreen(Screen)`** — replaces `ResultsTab`; contains `ScrollView` with dynamically built group cards
- **`FileChooserPopup(Popup)`** — replaces Flet `FilePicker`; modal popup with `FileChooserIconView` and Select/Cancel buttons
- **`ProgressPopup(Popup)`** — replaces `ft.AlertDialog` progress dialog; modal popup with `ProgressBar`, stage label, file label

Removed classes:
- `SearchTab(ft.Tab)` — replaced by `SearchScreen`
- `ResultsTab(ft.Tab)` — replaced by `ResultsScreen`
- `StyledTabButton(ft.Container)` — replaced by Kivy's built-in `TabbedPanel` or `ScreenManager` with custom header buttons
- `TabsControl(ft.Container)` — replaced by `ScreenManager` + top navigation buttons in `app.kv`

[Dependencies]
Replace `flet` with `kivy`; all other dependencies remain the same.

- **Remove**: `flet` (pip uninstall flet)
- **Add**: `kivy` (`pip install kivy`)
- **Add**: `kivymd` (optional, for Material Design widgets — `pip install kivymd`) — can be used for nicer buttons/cards but is optional
- Existing deps unchanged: `torch`, `torchvision`, `transformers`, `pillow`, `numpy`, `scikit-learn`, `opencv-python`
- `install.txt` updated accordingly

[Testing]
Manual testing of all UI interactions; no automated tests exist in the project.

- Launch `python app.py` and verify the window opens
- Test Browse button opens folder chooser and populates the path field
- Test threshold sliders update displayed values
- Test Find Duplicates triggers progress popup and background search
- Test results appear as cards with thumbnails after search completes
- Test double-click on thumbnail opens system image viewer
- Test config persistence: last directory reloaded on next launch
- Test with a folder of known duplicates to confirm end-to-end correctness

[Implementation Order]
Implement bottom-up: dependencies first, then widgets, then screens, then wiring.

1. Update `install.txt` and `README.md` to reference Kivy
2. Create `app.kv` with layout definitions for `SearchScreen`, `ResultsScreen`, `FileChooserPopup`, `ProgressPopup`, and the top navigation bar
3. Write `app.py` skeleton: `DinoDuplicateApp`, `SearchScreen`, `ResultsScreen` class stubs
4. Implement `FileChooserPopup` and `ProgressPopup` popup classes
5. Implement `SearchScreen` logic: browse, slider callbacks, `on_find_duplicates`, background thread + `Clock` progress updates
6. Implement `ResultsScreen.update_results` and `_build_group_card` with image thumbnails
7. Wire navigation: switching from Search to Results after search completes
8. Implement config persistence (load last directory on start, save on successful search)
9. Test full end-to-end flow and fix any Kivy-specific issues (texture loading, Unicode paths, thread safety)