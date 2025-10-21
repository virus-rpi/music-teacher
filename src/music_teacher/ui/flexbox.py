from typing import Optional, Union
import pygame
from pygame_gui import UIManager
from pygame_gui.core import UIContainer, UIElement
from pygame_gui.core.gui_type_hints import RectLike


class FlexBox(UIContainer):
    def __init__(
        self,
        manager: UIManager,
        relative_rect: RectLike,
        margin: int = 0,
        padding: int = 0,
        gap: int = 0,
        direction: str = "horizontal",
        align_x: str = "start",
        align_y: str = "center",
    ):
        sw, sh = manager.window_resolution
        super().__init__(
            pygame.Rect(
                relative_rect.x + padding,
                relative_rect.y + padding,
                sw - relative_rect.x,
                sh - relative_rect.y,
            ),
            manager,
        )
        self.width_prop = relative_rect.width if relative_rect.width > 0 else None
        self.height_prop = relative_rect.height if relative_rect.height > 0 else None
        self.margin = margin
        self.padding = padding
        self.gap = gap
        self.direction = direction  # 'horizontal' or 'vertical'
        self.align_x = align_x  # 'start', 'center', 'end'
        self.align_y = align_y  # 'start', 'center', 'end'
        self.element_data: list[
            tuple[
                UIElement,
                Union[Optional[float], str],
                Union[Optional[float], str],
                Optional[int],
                Optional[int],
            ]
        ] = []

    def place_element(
        self,
        element: UIElement,
        width_percent: Union[Optional[float], str] = None,
        height_percent: Union[Optional[float], str] = None,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
    ):
        """
        Add a pygame_gui element to the flexbox.
        If width or height is not set, only allow absolute pixel sizes for that dimension.
        Use "max" for width_percent or height_percent to automatically use maximum available space.
        """
        if (
            self.width_prop is None
            and width_percent is not None
            and width_percent != "max"
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use width_percent when FlexBox width is None. Use width_px instead."
            )
        if (
            self.height_prop is None
            and height_percent is not None
            and height_percent != "max"
            and not isinstance(element, FlexBox)
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use height_percent when FlexBox height is None. Use height_px instead."
            )
        element.set_container(self)
        self.element_data.append(
            (element, width_percent, height_percent, width_px, height_px)
        )
        self.rebuild_sizes()

    def insert_element(
        self,
        index: int,
        element: UIElement,
        width_percent: Union[Optional[float], str] = None,
        height_percent: Union[Optional[float], str] = None,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
    ):
        if (
            self.width_prop is None
            and width_percent is not None
            and width_percent != "max"
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use width_percent when FlexBox width is None. Use width_px instead."
            )
        if (
            self.height_prop is None
            and height_percent is not None
            and height_percent != "max"
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use height_percent when FlexBox height is None. Use height_px instead."
            )
        element.set_container(self)
        self.element_data.insert(
            index, (element, width_percent, height_percent, width_px, height_px)
        )
        self.rebuild_sizes()

    def replace_element(
        self,
        index: int,
        element: UIElement,
        width_percent: Union[Optional[float], str] = None,
        height_percent: Union[Optional[float], str] = None,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
    ):
        if (
            self.width_prop is None
            and width_percent is not None
            and width_percent != "max"
            and self.element_data[index][1] is not None
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use width_percent when FlexBox width is None. Use width_px instead."
            )
        if (
            self.height_prop is None
            and height_percent is not None
            and height_percent != "max"
            and self.element_data[index][2] is not None
            and not isinstance(self.ui_container, FlexBox)
        ):
            raise ValueError(
                "Cannot use height_percent when FlexBox height is None. Use height_px instead."
            )
        self.element_data[index][0].kill()
        element.set_container(self)
        self.element_data[index] = (
            element,
            width_percent or self.element_data[index][1],
            height_percent or self.element_data[index][2],
            width_px or self.element_data[index][3],
            height_px or self.element_data[index][4],
        )
        self.rebuild_sizes()

    def rebuild_sizes(self):
        n = len(self.element_data)
        if n == 0:
            return
        width, height = self.width_prop, self.height_prop
        if self.direction == "horizontal":
            if width is None and not isinstance(self.ui_container, FlexBox):
                total_width = sum(
                    wpx if wpx is not None else 0
                    for _, _, _, wpx, _ in self.element_data
                )
                total_width += self.gap * (n - 1) + 2 * self.margin
                width = total_width
            elif width is None:
                width = self.relative_rect[2]
            if height is None and not isinstance(self.ui_container, FlexBox):
                max_height = max(
                    hpx if hpx is not None else 1
                    for _, _, _, _, hpx in self.element_data
                )
                height = max_height + 2 * self.margin
            elif height is None:
                height = self.relative_rect[3]
        else:  # vertical
            if height is None and not isinstance(self.ui_container, FlexBox):
                total_height = sum(
                    hpx if hpx is not None else 0
                    for _, _, _, _, hpx in self.element_data
                )
                total_height += self.gap * (n - 1) + 2 * self.margin
                height = total_height
            elif height is None:
                height = self.relative_rect[3]
            if width is None and not isinstance(self.ui_container, FlexBox):
                max_width = max(
                    wpx if wpx is not None else 1
                    for _, _, _, wpx, _ in self.element_data
                )
                width = max_width + 2 * self.margin
            elif width is None:
                width = self.relative_rect[2]

        width -= 2 * self.padding
        height -= 2 * self.padding

        if self.direction == "horizontal":
            total_specified = sum(
                w for _, w, _, _, _ in self.element_data if w is not None and w != "max"
            )
            max_width_elements = [
                i for i, (_, w, _, _, _) in enumerate(self.element_data) if w == "max"
            ]
            unspecified = [
                i for i, (_, w, _, _, _) in enumerate(self.element_data) if w is None
            ]
            available_content_width = (
                width - 2 * self.margin - self.gap * (n - 1) if width is not None else 0
            )
            used_width = 0
            if width is not None:
                used_width += total_specified * available_content_width
            pixel_width = sum(
                wpx
                for _, w, _, wpx, _ in self.element_data
                if wpx is not None and w != "max"
            )
            used_width += pixel_width
            remaining_width = (
                available_content_width - used_width if width is not None else 0
            )
            max_width_share = (
                remaining_width / len(max_width_elements)
                if max_width_elements and remaining_width > 0
                else 0
            )
            unspecified_share = (
                remaining_width / len(unspecified)
                if unspecified and remaining_width > 0 and not max_width_elements
                else 0
            )

            x = self.margin
            for i, (elem, w, h, wpx, hpx) in enumerate(self.element_data):
                if width is not None:
                    if w == "max":
                        ew = int(max_width_share)
                    elif w is not None:
                        ew = int(w * available_content_width)
                    elif wpx is not None:
                        ew = wpx
                    else:
                        ew = int(unspecified_share)
                else:
                    ew = wpx

                if height is not None:
                    if h == "max":
                        eh = height - 2 * self.margin
                    elif h is not None:
                        eh = int(h * (height - 2 * self.margin))
                    else:
                        eh = hpx
                else:
                    eh = hpx

                if self.align_y == "start":
                    y = self.margin
                elif self.align_y == "center":
                    y = self.margin + (
                        (height - 2 * self.margin - eh) // 2
                        if height is not None
                        else 0
                    )
                else:  # 'end'
                    y = height - self.margin - eh if height is not None else self.margin

                elem.set_relative_position((x, y))
                elem.set_dimensions((ew, eh))
                x += ew + self.gap
        else:  # vertical
            total_specified = sum(
                h for _, _, h, _, _ in self.element_data if h is not None and h != "max"
            )
            max_height_elements = [
                i for i, (_, _, h, _, _) in enumerate(self.element_data) if h == "max"
            ]
            unspecified = [
                i for i, (_, _, h, _, _) in enumerate(self.element_data) if h is None
            ]
            available_content_height = (
                height - 2 * self.margin - self.gap * (n - 1)
                if height is not None
                else 0
            )
            used_height = 0
            if height is not None:
                used_height += total_specified * available_content_height
            pixel_height = sum(
                hpx
                for _, _, h, _, hpx in self.element_data
                if hpx is not None and h != "max"
            )
            used_height += pixel_height
            remaining_height = (
                available_content_height - used_height if height is not None else 0
            )
            max_height_share = (
                remaining_height / len(max_height_elements)
                if max_height_elements and remaining_height > 0
                else 0
            )
            unspecified_share = (
                remaining_height / len(unspecified)
                if unspecified and remaining_height > 0 and not max_height_elements
                else 0
            )

            y = self.margin
            for i, (elem, w, h, wpx, hpx) in enumerate(self.element_data):
                if height is not None:
                    if h == "max":
                        eh = int(max_height_share)
                    elif h is not None:
                        eh = int(h * available_content_height)
                    elif hpx is not None:
                        eh = hpx
                    else:
                        eh = int(unspecified_share)
                else:
                    eh = hpx

                if width is not None:
                    if w == "max":
                        ew = width - 2 * self.margin
                    elif w is not None:
                        ew = int(w * (width - 2 * self.margin))
                    else:
                        ew = wpx
                else:
                    ew = wpx

                if self.align_x == "start":
                    x = self.margin
                elif self.align_x == "center":
                    x = self.margin + (
                        (width - 2 * self.margin - ew) // 2 if width is not None else 0
                    )
                else:  # 'end'
                    x = width - self.margin - ew if width is not None else self.margin

                elem.set_relative_position((x, y))
                elem.set_dimensions((ew, eh))
                y += eh + self.gap
