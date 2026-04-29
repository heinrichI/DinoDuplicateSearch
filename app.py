"""
Flet Desktop Application for Finding Duplicate Images
"""
import flet as ft
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from duplicates_finder import DuplicatesFinder, DuplicateGroup

# Config file for persistent settings
CONFIG_FILE = "config.json"


def load_config() -> dict:
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_config(config: dict):
    """Save configuration to file"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

# Icon shortcuts for Flet 0.84.0
ICONS = ft.icons.Icons


class SearchTab(ft.Tab):
    """Search Tab with content"""
    
    def __init__(self, finder: DuplicatesFinder, on_results_ready, switch_tabs_callback=None):
        # Initialize flet Tab first
        super().__init__(
            label="Search",
            icon=ICONS.SEARCH_OUTLINED,
        )
        
        # Then set our custom attributes
        self.finder = finder
        self.on_results_ready = on_results_ready
        self.switch_tabs_callback = switch_tabs_callback
        self.results = []
        
        # Queue for thread-safe progress updates
        self._progress_queue = asyncio.Queue()
        
        # Initialize UI elements
        self.geometric_check = ft.Switch(
            label="Enable geometric verification (SIFT/WGC)",
            value=True
        )
        self.wgc_threshold_text = ft.Text("0.30")
        self.wgc_threshold_slider = ft.Slider(
            min=0.1,
            max=0.9,
            value=0.30,
            divisions=16,
            on_change=self._on_wgc_threshold_change,
        )
        self.file_picker = ft.FilePicker()
        self.directory_input = ft.TextField(
            label="Directory Path",
            hint_text="Enter folder path or use Browse button...",
            expand=True
        )
        self.threshold_value_text = ft.Text("0.45")
        self.threshold_slider = ft.Slider(
            min=0.01,
            max=1.0,
            value=0.45,
            divisions=99,
            on_change=self._on_threshold_change,
        )
        
        # Progress dialog with detailed status
        self.progress_bar_for_dialog = ft.ProgressBar(width=300)
        self.progress_stage_text = ft.Text("Initializing...", size=12)
        self.progress_file_text = ft.Text("", size=10, italic=True)
        self.progress_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Searching...", weight=ft.FontWeight.BOLD),
            content=ft.Column([
                self.progress_bar_for_dialog,
                self.progress_stage_text,
                self.progress_file_text,
            ], spacing=5),
            actions=[],
        )
        
        # Find Duplicates button - store reference to disable during search
        self.find_duplicates_button = ft.Button(
            "Find Duplicates",
            icon=ICONS.SEARCH_OUTLINED,
            on_click=self._on_find_duplicates_click
        )
        
        self.content = self._build_content()
        
        # Load last directory from config
        config = load_config()
        if config.get("last_directory"):
            self.directory_input.value = config["last_directory"]
        
    def _on_threshold_change(self, e):
        self.threshold_value_text.value = f"{e.control.value:.2f}"
        # Note: No need to call self.page.update() here - the Text updates automatically
        # when placed in a Container that is already on the page
    
    def _on_wgc_threshold_change(self, e):
        self.wgc_threshold_text.value = f"{e.control.value:.2f}"
    
    def _build_content(self):
        return ft.Container(
            content=ft.Column([
                ft.Text("Select Directory", size=20, weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.directory_input,
                    ft.Button(
                        "Browse...",
                        icon=ICONS.FOLDER_OPEN_OUTLINED,
                        on_click=self._on_browse_click
                    ),
                ]),
                ft.Divider(),
                ft.Text("Settings", size=20, weight=ft.FontWeight.BOLD),
                ft.Text("Distance Threshold (lower = stricter):"),
                ft.Row([self.threshold_slider, self.threshold_value_text]),
                self.geometric_check,
                ft.Row([self.wgc_threshold_slider, self.wgc_threshold_text]),
                ft.Divider(),
                # Use stored button reference
                self.find_duplicates_button,
            ], spacing=10),
            padding=20,
        )
    
    def _get_page(self):
        """Safely get page reference"""
        try:
            return self.page if hasattr(self, 'page') else None
        except RuntimeError:
            return None
    
    def _update_page(self):
        page = self._get_page()
        if page:
            page.update()
    
    async def _on_browse_click(self, e):
        result = await self.file_picker.get_directory_path()
        if result:
            self.directory_input.value = result
            self._update_page()
    
    async def _on_find_duplicates_click(self, e):
        folder = self.directory_input.value
        if not folder or not os.path.isdir(folder):
            self._show_error("Please select a valid directory")
            return
        
        # Get page reference from event
        page = e.control.page if e.control else None
        
        # Show progress dialog
        if page:
            page.show_dialog(self.progress_dialog)
            page.update()
        
        # Disable button during search
        self.find_duplicates_button.disabled = True
        if page:
            page.update()
        
        # Clear and prepare queue
        while not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Flag to track if search is done
        search_done = False
        search_results = None
        search_error = None
        
        def progress_callback(percent, message):
            """Called from background thread - put update in queue"""
            try:
                self._progress_queue.put_nowait((percent, message))
            except Exception as ex:
                print(f"Queue put error: {ex}")
        
        async def progress_updater():
            """Async task that processes progress updates from queue"""
            nonlocal search_done
            while not search_done:
                try:
                    # Wait for update with timeout
                    percent, message = await asyncio.wait_for(
                        self._progress_queue.get(),
                        timeout=0.1
                    )
                    
                    # Update UI elements
                    self.progress_bar_for_dialog.value = percent / 100
                    self.progress_stage_text.value = message.split('\n')[0] if message else "Processing..."
                    if '\n' in message:
                        self.progress_file_text.value = message.split('\n', 1)[1]
                    else:
                        self.progress_file_text.value = ""
                    
                    # Update page
                    if page:
                        page.update()
                        
                except asyncio.TimeoutError:
                    # No update in queue, check if search is done
                    continue
                except Exception as ex:
                    print(f"Progress update error: {ex}")
        
        async def run_search():
            """Run search in executor"""
            nonlocal search_done, search_results, search_error
            try:
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(
                    ThreadPoolExecutor(),
                    lambda: self.finder.find_duplicates(
                        folder_path=folder,
                        distance_threshold=self.threshold_slider.value,
                        enable_geometric_check=self.geometric_check.value,
                        wgc_threshold=self.wgc_threshold_slider.value,
                        progress_callback=progress_callback
                    )
                )
            except Exception as ex:
                search_error = ex
            finally:
                search_done = True
        
        try:
            # Run both tasks concurrently
            await asyncio.gather(
                run_search(),
                progress_updater()
            )
            
            # Re-enable button
            self.find_duplicates_button.disabled = False
            
            # Close progress dialog
            if page:
                page.pop_dialog()
                page.update()
            
            # Handle errors
            if search_error:
                self._show_error(f"Error: {str(search_error)}")
                return
            
            # Save last directory
            config = load_config()
            config["last_directory"] = folder
            save_config(config)
            
            # Switch to results tab
            if self.switch_tabs_callback:
                self.switch_tabs_callback()
            
            if page:
                page.update()
            
            # Small delay then show results
            await asyncio.sleep(0.1)
            self.on_results_ready(search_results or [])
            
            if page:
                page.update()
            
        except Exception as ex:
            self.find_duplicates_button.disabled = False
            if page:
                page.pop_dialog()
                page.update()
            self._show_error(f"Error: {str(ex)}")
    
    def _show_error(self, message: str):
        page = self._get_page()
        if page:
            page.show_snack_bar(ft.SnackBar(content=ft.Text(message)))
    
    def did_mount(self):
        self.page.overlay.append(self.file_picker)
        # Note: progress_dialog is added in main() to avoid duplicates


class ResultsTab(ft.Tab):
    """Results Tab with duplicate groups"""
    
    def __init__(self):
        self.results = []
        self.results_column = ft.ListView(expand=True, spacing=10)
        self.count_text = ft.Text("Found: 0 groups")
        
        super().__init__(
            label="Results",
            icon=ICONS.LIST_OUTLINED,
        )
        
        self.content = self._build_content()
        
        # Show placeholder initially
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message when no results"""
        self.results_column.controls.clear()
        self.results_column.controls.append(
            ft.Container(
                content=ft.Column([
                    ft.Icon(ICONS.SEARCH_OFF, size=48),
                    ft.Text("Run search in Search tab first", size=16),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.Alignment(0, 0),
                padding=50,
            )
        )
        self.count_text.value = "Found: 0 groups"
        try:
            if self.page:
                self.page.update()
        except:
            pass
    
    def _build_content(self):
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Duplicate Groups", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    self.count_text,
                ]),
                ft.Divider(),
                self.results_column,
            ], spacing=10),
            padding=20,
        )
    
    def update_results(self, results: list[DuplicateGroup]):
        self.results = results
        self.results_column.controls.clear()
        
        if not results:
            self.results_column.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ICONS.SEARCH_OFF, size=48),
                        ft.Text("No duplicates found", size=16),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    alignment=ft.Alignment(0, 0),
                    padding=50,
                )
            )
        else:
            for i, group in enumerate(results):
                self.results_column.controls.append(
                    self._create_group_card(group, i)
                )
        
        self.count_text.value = f"Found: {len(results)} groups"
        self.page.update() if self.page else None
    
    def _create_group_card(self, group: DuplicateGroup, index: int) -> ft.Card:
        """Create a card widget for a duplicate group"""
        paths = group.paths
        
        # Group header with geometric info
        geo_subtitle = f"Avg similarity: {group.avg_similarity:.4f}"
        if hasattr(group, 'is_geometric_verified') and group.is_geometric_verified:
            # Get votes from first verified pair
            for pair in group.pairs:
                if pair.geometric_verified:
                    geo_subtitle = f"∠ {pair.geometric_angle:.1f}° ({pair.geometric_angle_votes} votes)  × {pair.geometric_scale:.2f} ({pair.geometric_scale_votes} votes)"
                    break
        
        header = ft.ListTile(
            leading=ft.Icon(ICONS.FOLDER_OUTLINED),
            title=ft.Text(f"Group {index + 1} ({len(paths)} images)"),
            subtitle=ft.Text(geo_subtitle),
        )
        
        # Get unique image paths
        image_paths = []
        for pair in group.pairs:
            for path in [pair.path1, pair.path2]:
                if path not in image_paths:
                    image_paths.append(path)
        
        # Create GridView for thumbnails
        thumbnails = ft.GridView(
            expand=True,
            runs_count=min(4, len(image_paths)),
            max_extent=120,
            child_aspect_ratio=1.0,
            spacing=5,
            run_spacing=5,
        )
        
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            thumbnails.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Image(
                            src=img_path,
                            border_radius=ft.BorderRadius.all(8),
                            width=100,
                            height=100,
                        ),
                        ft.Text(basename[:20], size=8, tooltip=basename),
                    ], spacing=2),
                    alignment=ft.Alignment(0, 0),
                )
            )
        
        tile_content = ft.Column([
            header,
            ft.Divider(),
            ft.Container(
                content=thumbnails,
                height=min(250, 60 + len(image_paths) * 65),
            ),
        ], spacing=5)
        
        return ft.Card(content=tile_content, elevation=2)


class StyledTabButton(ft.Container):
    """A tab button with underline styling"""
    
    def __init__(self, label: str, icon: str, is_selected: bool, on_click):
        self.label = label
        self.is_selected = is_selected
        self.on_click_handler = on_click
        
        # Border color based on selection
        border_color = ft.Colors.PRIMARY if is_selected else ft.Colors.TRANSPARENT
        
        super().__init__(
            content=ft.Row([
                ft.Icon(icon, size=18),
                ft.Text(label, size=14, weight=ft.FontWeight.W_500),
            ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
            padding=ft.Padding(left=16, top=12, right=16, bottom=12),
            border=ft.Border.only(bottom=ft.border.BorderSide(2, border_color)),
            on_click=self._handle_click,
        )
    
    def _handle_click(self, e):
        if self.on_click_handler:
            self.on_click_handler()
    
    def set_selected(self, selected: bool):
        self.is_selected = selected
        border_color = ft.Colors.PRIMARY if selected else ft.Colors.TRANSPARENT
        self.border = ft.Border.only(bottom=ft.border.BorderSide(2, border_color))
        try:
            if self.page:
                self.update()
        except RuntimeError:
            pass


class TabsControl(ft.Container):
    """Custom tabs implementation with underline styling"""
    
    def __init__(self, tabs_list: list[ft.Tab], on_change=None):
        self.tabs_list = tabs_list
        self.on_change = on_change
        self.selected_index = 0
        self.tab_buttons: list[StyledTabButton] = []
        
        # Store reference for switching tabs from outside
        self._switch_to_results = lambda: self._select_tab(1)
        
        # Build tab buttons
        self.tab_bar = ft.Container(
            content=ft.Row(
                controls=self._build_tab_buttons(),
                spacing=0,
            ),
            border=ft.Border.only(bottom=ft.border.BorderSide(1, ft.Colors.ON_SURFACE_VARIANT)),
        )
        
        # Content area - delay access until tabs are ready
        self.content_area = ft.Container(expand=True)
        
        super().__init__(
            content=ft.Column([
                self.tab_bar,
                self.content_area,
            ], spacing=0),
            expand=True,
        )
        
        # Set initial content after super().__init__ is called
        first_tab = next((t for t in self.tabs_list if t is not None), None)
        if first_tab:
            self.content_area.content = first_tab.content
    
    def _build_tab_buttons(self) -> list:
        buttons = []
        for i, tab in enumerate(self.tabs_list):
            if tab is None:
                continue
            btn = StyledTabButton(
                label=tab.label or f"Tab {i+1}",
                icon=tab.icon,
                is_selected=(i == self.selected_index),
                on_click=lambda idx=i: self._select_tab(idx)
            )
            self.tab_buttons.append(btn)
            buttons.append(btn)
        return buttons
    
    def _select_tab(self, index: int):
        if index >= len(self.tabs_list) or self.tabs_list[index] is None:
            return
        self.selected_index = index
        self.content_area.content = self.tabs_list[index].content
        
        # Update button styles
        for i, btn in enumerate(self.tab_buttons):
            btn.set_selected(i == index)
        
        if self.on_change:
            self.on_change(index)


def main(page: ft.Page):
    page.title = "DINOv2 Duplicate Finder"
    page.window_width = 1000
    page.window_height = 700
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # State
    finder = DuplicatesFinder()
    results_tab = ResultsTab()
    
    def on_results_ready(results):
        results_tab.update_results(results)
    
    # Create search tab first (with reference to tabs_control)
    tabs_control = None  # Will be set after creation
    
    search_tab = SearchTab(finder, on_results_ready=on_results_ready, switch_tabs_callback=lambda: tabs_control._select_tab(1) if tabs_control else None)
    
    # Add progress dialog to page overlay
    page.overlay.append(search_tab.progress_dialog)
    
    # Custom tabs control with both tabs
    tabs_control = TabsControl([search_tab, results_tab])
    
    page.add(tabs_control)
    page.update()


if __name__ == "__main__":
    ft.run(main)
