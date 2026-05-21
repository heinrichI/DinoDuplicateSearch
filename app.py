"""
Kivy Desktop Application for Finding Duplicate Images
"""
import os
import json
import threading
import traceback
from functools import partial

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.image import Image as KivyImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.core.window import Window

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


def open_original_image(path: str):
    """Open image in system default viewer"""
    try:
        os.startfile(path)
    except Exception as e:
        print(f"Error opening file: {e}")


class FileChooserPopup(Popup):
    """Popup with file chooser for selecting a directory"""
    current_path = StringProperty(os.path.expanduser("~"))
    selected_path = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load last directory from config as starting point
        config = load_config()
        if config.get("last_directory"):
            self.current_path = config["last_directory"]

    def on_select(self):
        """Called when user clicks Select"""
        file_chooser = self.ids.file_chooser
        if file_chooser.path and file_chooser.selection:
            selected = file_chooser.selection[0]
            if os.path.isdir(selected):
                self.selected_path = selected
            else:
                self.selected_path = file_chooser.path
        else:
            self.selected_path = file_chooser.path
        self.dismiss()


class ProgressPopup(Popup):
    """Popup with progress bar for long operations"""
    pass


class SearchScreen(Screen):
    """Search screen with directory picker, settings, and find button"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.finder = DuplicatesFinder()
        self.search_thread = None
        self._progress_popup = None
        # Load last directory into the input field
        config = load_config()
        last_dir = config.get("last_directory")
        if last_dir:
            Clock.schedule_once(lambda dt, d=last_dir: setattr(self.ids.directory_input, 'text', d))

    def on_threshold_change(self, value):
        """Update threshold label when slider changes"""
        label = self.ids.threshold_value_label
        label.text = f"{value:.2f}"

    def on_wgc_threshold_change(self, value):
        """Update WGC threshold label when slider changes"""
        label = self.ids.wgc_threshold_value_label
        label.text = f"{value:.2f}"

    def on_browse_click(self):
        """Open file chooser popup"""
        popup = FileChooserPopup()
        popup.bind(on_dismiss=lambda instance: self._on_folder_selected(instance.selected_path))
        popup.open()

    def _on_folder_selected(self, path):
        """Called when a folder is selected in the file chooser"""
        if path:
            self.ids.directory_input.text = path

    def switch_to_results(self):
        """Switch to results screen"""
        self.manager.current = 'results'

    def on_find_duplicates(self):
        """Start duplicate search in background thread"""
        folder = self.ids.directory_input.text.strip()
        if not folder or not os.path.isdir(folder):
            self._show_error("Please select a valid directory")
            return

        threshold = self.ids.threshold_slider.value
        geo_check = self.ids.geometric_check.active
        wgc_threshold = self.ids.wgc_threshold_slider.value

        # Disable button during search
        self.ids.find_button.disabled = True
        self.ids.find_button.text = "Searching..."

        # Show progress popup
        self._progress_popup = ProgressPopup()
        self._progress_popup.open()

        # Reset progress
        self._progress_popup.ids.progress_bar.value = 0.0
        self._progress_popup.ids.stage_label.text = "Initializing..."
        self._progress_popup.ids.file_label.text = ""

        # Run search in background thread
        self.search_thread = threading.Thread(
            target=self._run_search_thread,
            args=(folder, threshold, geo_check, wgc_threshold),
            daemon=True
        )
        self.search_thread.start()

    def _run_search_thread(self, folder, threshold, geo_check, wgc_threshold):
        """Run search in background thread"""
        results = []
        error = None
        try:
            # Define progress callback that schedules UI updates
            def progress_callback(percent, message):
                Clock.schedule_once(
                    lambda dt, p=percent, m=message: self._update_progress(p, m),
                    0
                )

            results = self.finder.find_duplicates(
                folder_path=folder,
                distance_threshold=threshold,
                enable_geometric_check=geo_check,
                wgc_threshold=wgc_threshold,
                progress_callback=progress_callback
            )
        except Exception as ex:
            error = ex
            traceback.print_exc()

        # Schedule completion on main thread
        Clock.schedule_once(
            lambda dt, r=results, e=error: self._on_search_complete(r, e, folder),
            0
        )

    def _update_progress(self, percent, message):
        """Update progress popup from main thread"""
        if self._progress_popup is None:
            return
        popup = self._progress_popup
        popup.ids.progress_bar.value = float(percent)

        if '\n' in message:
            lines = message.split('\n', 1)
            popup.ids.stage_label.text = lines[0]
            popup.ids.file_label.text = lines[1] if len(lines) > 1 else ""
        else:
            popup.ids.stage_label.text = message
            popup.ids.file_label.text = ""

    def _on_search_complete(self, results, error, folder):
        """Called when search completes"""
        # Re-enable button
        self.ids.find_button.disabled = False
        self.ids.find_button.text = "Find Duplicates"

        # Dismiss progress popup
        if self._progress_popup:
            self._progress_popup.dismiss()
            self._progress_popup = None

        # Handle errors
        if error:
            self._show_error(f"Error: {str(error)}")
            return

        # Save last directory to config
        config = load_config()
        config["last_directory"] = folder
        save_config(config)

        # Send results to results screen
        results_screen = self.manager.get_screen('results')
        results_screen.update_results(results)

        # Switch to results tab
        self.manager.current = 'results'

    def _show_error(self, message: str):
        """Show error message in a popup"""
        popup = Popup(
            title="Error",
            content=Label(text=message),
            size_hint=(0.6, 0.3)
        )
        popup.open()


class GroupCard(BoxLayout):
    """A card widget for a duplicate group"""
    pass


class ResultsScreen(Screen):
    """Results screen with duplicate groups"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_cards = []

    def switch_to_search(self):
        """Switch to search screen"""
        self.manager.current = 'search'

    def update_results(self, groups):
        """Update the results display with new duplicate groups"""
        grid = self.ids.results_grid
        grid.clear_widgets()
        self.current_cards.clear()

        self.ids.count_label.text = f"Found: {len(groups)} groups"

        if not groups:
            grid.add_widget(
                BoxLayout(
                    orientation='vertical',
                    size_hint_y=None,
                    height=200,
                    padding=50
                )
            )
            return

        for i, group in enumerate(groups):
            card = self._build_group_card(group, i)
            self.current_cards.append(card)
            grid.add_widget(card)

    def _build_group_card(self, group: DuplicateGroup, index: int) -> BoxLayout:
        """Create a card widget for a duplicate group"""
        paths = group.paths

        # Build subtitle text
        geo_subtitle = f"Avg similarity: {group.avg_similarity:.4f}"
        if hasattr(group, 'is_geometric_verified') and group.is_geometric_verified:
            for pair in group.pairs:
                if pair.geometric_verified:
                    geo_subtitle = (
                        f"Angle: {pair.geometric_angle:.1f}° "
                        f"({pair.geometric_angle_votes} votes)  "
                        f"Scale: {pair.geometric_scale:.2f} "
                        f"({pair.geometric_scale_votes} votes)"
                    )
                    break

        # Get unique image paths
        image_paths = []
        for pair in group.pairs:
            for path in [pair.path1, pair.path2]:
                if path not in image_paths:
                    image_paths.append(path)

        # Main card container
        card = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=self._calculate_card_height(len(image_paths)),
            padding=10,
            spacing=5
        )

        # Header
        header = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=30
        )
        header.add_widget(Label(
            text=f"Group {index + 1} ({len(paths)} images)",
            bold=True,
            font_size=15,
            halign='left',
            valign='middle'
        ))
        header.add_widget(Label(
            text=geo_subtitle,
            font_size=12,
            halign='right',
            valign='middle'
        ))
        card.add_widget(header)

        # Separator
        sep = BoxLayout(size_hint_y=None, height=2)
        sep.canvas.before.clear()
        with sep.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.7, 0.7, 0.7, 1)
            Rectangle(pos=sep.pos, size=sep.size)
        sep.bind(pos=lambda instance, value: self._update_separator(instance))
        sep.bind(size=lambda instance, value: self._update_separator(instance))
        card.add_widget(sep)

        # Thumbnails grid
        thumb_grid = GridLayout(
            cols=min(4, len(image_paths)),
            spacing=5,
            size_hint_y=None,
            height=self._calculate_thumbnails_height(len(image_paths))
        )

        for img_path in image_paths:
            basename = os.path.basename(img_path)
            # Container for one thumbnail
            thumb_container = BoxLayout(
                orientation='vertical',
                size_hint=(1, None),
                height=160,
                spacing=2
            )

            # Image widget
            try:
                img = KivyImage(
                    source=img_path,
                    size_hint=(1, 0.8),
                    allow_stretch=True,
                    keep_ratio=True
                )
            except Exception:
                img = Label(text="[No preview]", size_hint=(1, 0.8))

            # Double-tap detection
            img.bind(on_touch_down=partial(self._on_thumbnail_touch, img_path))

            # Filename label
            name_label = Label(
                text=basename[:20] + ('...' if len(basename) > 20 else ''),
                font_size=10,
                size_hint=(1, 0.2),
                text_size=(140, None),
                halign='center'
            )

            thumb_container.add_widget(img)
            thumb_container.add_widget(name_label)
            thumb_grid.add_widget(thumb_container)

        card.add_widget(thumb_grid)

        # Card border using canvas
        card.canvas.before.clear()
        with card.canvas.before:
            from kivy.graphics import Color, Rectangle, Line
            Color(0.9, 0.9, 0.9, 1)
            Rectangle(pos=card.pos, size=card.size)
            Color(0.6, 0.6, 0.6, 1)
            Line(rectangle=(card.x, card.y, card.width, card.height), width=1)
        card.bind(pos=lambda instance, value: self._redraw_card_border(instance))
        card.bind(size=lambda instance, value: self._redraw_card_border(instance))

        return card

    def _calculate_card_height(self, num_paths):
        """Calculate card height based on number of images"""
        if num_paths <= 4:
            return 220
        elif num_paths <= 8:
            return 380
        else:
            return 540

    def _calculate_thumbnails_height(self, num_paths):
        """Calculate thumbnails area height"""
        rows = (num_paths + 3) // 4  # ceil division
        return rows * 170

    def _on_thumbnail_touch(self, img_path, instance, touch):
        """Handle double-tap on thumbnail to open original image"""
        if instance.collide_point(*touch.pos) and touch.is_double_tap:
            open_original_image(img_path)

    def _update_separator(self, instance):
        """Redraw separator line"""
        instance.canvas.before.clear()
        with instance.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.7, 0.7, 0.7, 1)
            Rectangle(pos=instance.pos, size=instance.size)

    def _redraw_card_border(self, instance):
        """Redraw card background and border"""
        instance.canvas.before.clear()
        with instance.canvas.before:
            from kivy.graphics import Color, Rectangle, Line
            Color(0.95, 0.95, 0.95, 1)
            Rectangle(pos=instance.pos, size=instance.size)
            Color(0.6, 0.6, 0.6, 1)
            Line(rectangle=(instance.x, instance.y, instance.width, instance.height), width=1)


class DinoDuplicateApp(App):
    """Main Kivy application"""
    title = "DINOv2 Duplicate Finder"

    def build(self):
        # Window size
        Window.size = (1000, 700)
        # Load KV rules for all custom widget classes
        Builder.load_file('app.kv')
        # Build screen manager with screens
        sm = ScreenManager()
        sm.add_widget(SearchScreen(name='search'))
        sm.add_widget(ResultsScreen(name='results'))
        return sm


if __name__ == "__main__":
    DinoDuplicateApp().run()