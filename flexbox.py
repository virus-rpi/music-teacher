from typing import List, Optional, Tuple

import pygame
from pygame_gui import UIManager
from pygame_gui.core import UIContainer, UIElement
from pygame_gui.core.gui_type_hints import RectLike


class FlexBox(UIContainer):
    def __init__(self, manager: UIManager, relative_rect: RectLike, margin: int = 0, gap: int = 0, direction: str = 'horizontal'):
        sw, sh = manager.window_resolution
        super().__init__(pygame.Rect(relative_rect.x, relative_rect.y, sw-relative_rect.x, sh-relative_rect.y), manager)
        self.width_prop = relative_rect.width if relative_rect.width > 0 else None
        self.height_prop = relative_rect.height if relative_rect.height > 0 else None
        self.margin = margin
        self.gap = gap
        self.direction = direction  # 'horizontal' or 'vertical'
        self.element_data: List[Tuple[UIElement, Optional[float], Optional[float], Optional[int], Optional[int]]] = []

    def place_element(self, element: UIElement, width_percent: Optional[float] = None, height_percent: Optional[float] = None, width_px: Optional[int] = None, height_px: Optional[int] = None):
        """
        Add a pygame_gui element to the flexbox.
        If width or height is not set, only allow absolute pixel sizes for that dimension.
        """
        if self.width_prop is None and width_percent is not None and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use width_percent when FlexBox width is None. Use width_px instead.")
        if self.height_prop is None and height_percent is not None and not isinstance(element, FlexBox) and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use height_percent when FlexBox height is None. Use height_px instead.")
        element.set_container(self)
        self.element_data.append((element, width_percent, height_percent, width_px, height_px))
        self.rebuild_sizes()

    def insert_element(self, index: int, element: UIElement, width_percent: Optional[float] = None, height_percent: Optional[float] = None, width_px: Optional[int] = None, height_px: Optional[int] = None):
        if self.width_prop is None and width_percent is not None and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use width_percent when FlexBox width is None. Use width_px instead.")
        if self.height_prop is None and height_percent is not None and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use height_percent when FlexBox height is None. Use height_px instead.")
        element.set_container(self)
        self.element_data.insert(index, (element, width_percent, height_percent, width_px, height_px))
        self.rebuild_sizes()

    def replace_element(self, index: int, element: UIElement, width_percent: Optional[float] = None, height_percent: Optional[float] = None, width_px: Optional[int] = None, height_px: Optional[int] = None):
        if self.width_prop is None and width_percent is not None and self.element_data[index][1] is not None and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use width_percent when FlexBox width is None. Use width_px instead.")
        if self.height_prop is None and height_percent is not None and self.element_data[index][2] is not None and not isinstance(self.ui_container, FlexBox):
            raise ValueError("Cannot use height_percent when FlexBox height is None. Use height_px instead.")
        self.element_data[index][0].kill()
        element.set_container(self)
        self.element_data[index] = (element, width_percent or self.element_data[index][1], height_percent or self.element_data[index][2], width_px or self.element_data[index][3], height_px or self.element_data[index][4])
        self.rebuild_sizes()

    def rebuild_sizes(self):
        n = len(self.element_data)
        if n == 0:
            return self
        width, height = self.width_prop, self.height_prop
        if self.direction == 'horizontal':
            if width is None and not isinstance(self.ui_container, FlexBox):
                total_width = sum(wpx if wpx is not None else 0 for _, _, _, wpx, _ in self.element_data)
                total_width += self.gap * (n - 1) + 2 * self.margin
                width = total_width
            elif width is None:
                width = self.relative_rect[2]
            if height is None and not isinstance(self.ui_container, FlexBox):
                max_height = max(hpx if hpx is not None else 1 for _, _, _, _, hpx in self.element_data)
                height = max_height + 2 * self.margin
            elif height is None:
                height = self.relative_rect[3]
        else:  # vertical
            if height is None and not isinstance(self.ui_container, FlexBox):
                total_height = sum(hpx if hpx is not None else 0 for _, _, _, _, hpx in self.element_data)
                total_height += self.gap * (n - 1) + 2 * self.margin
                height = total_height
            elif height is None:
                height = self.relative_rect[3]
            if width is None and not isinstance(self.ui_container, FlexBox):
                max_width = max(wpx if wpx is not None else 1 for _, _, _, wpx, _ in self.element_data)
                width = max_width + 2 * self.margin
            elif width is None:
                height = self.relative_rect[2]
        if self.direction == 'horizontal':
            total_specified = sum(w for _, w, _, _, _ in self.element_data if w is not None)
            unspecified = [i for i, (_, w, _, _, _) in enumerate(self.element_data) if w is None]
            remaining = 1.0 - total_specified if width is not None else 0
            even_width = remaining / len(unspecified) if unspecified and width is not None else 0
            x = self.margin
            for i, (elem, w, h, wpx, hpx) in enumerate(self.element_data):
                if width is not None:
                    wp = w if w is not None else even_width
                    ew = int(wp * (width - 2 * self.margin - self.gap * (n - 1))) if w is not None else wpx
                else:
                    ew = wpx
                if height is not None:
                    hp = h if h is not None else 1.0
                    eh = int(hp * (height - 2 * self.margin)) if h is not None else hpx
                else:
                    eh = hpx
                y = self.margin + ((height - 2 * self.margin - eh) // 2 if height is not None else 0)
                elem.set_relative_position((x, y))
                elem.set_dimensions((ew, eh))
                x += ew + self.gap
        else:  # vertical
            total_specified = sum(h for _, _, h, _, _ in self.element_data if h is not None)
            unspecified = [i for i, (_, _, h, _, _) in enumerate(self.element_data) if h is None]
            remaining = 1.0 - total_specified if height is not None else 0
            even_height = remaining / len(unspecified) if unspecified and height is not None else 0
            y = self.margin
            for i, (elem, w, h, wpx, hpx) in enumerate(self.element_data):
                if height is not None:
                    hp = h if h is not None else even_height
                    eh = int(hp * (height - 2 * self.margin - self.gap * (n - 1))) if h is not None else hpx
                else:
                    eh = hpx
                if width is not None:
                    wp = w if w is not None else 1.0
                    ew = int(wp * (width - 2 * self.margin)) if w is not None else wpx
                else:
                    ew = wpx
                x = self.margin + ((width - 2 * self.margin - ew) // 2 if width is not None else 0)
                elem.set_relative_position((x, y))
                elem.set_dimensions((ew, eh))
                y += eh + self.gap
        return self
