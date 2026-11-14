"""Physical spacing helpers for shared labels on thesis figure panels."""

from __future__ import annotations

SHARED_YLABEL_GAP_PT = 6.0
SHARED_YLABEL_OUTER_PADDING_PT = 8.0


def _left_tick_boundary(fig, axes) -> tuple[float, float]:
    renderer = fig.canvas.get_renderer()
    left_spine = min(ax.bbox.x0 for ax in axes)
    boxes = []
    for ax in axes:
        if abs(ax.bbox.x0 - left_spine) > .5:
            continue
        low, high = sorted(ax.get_ylim())
        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            if low <= tick.get_loc() <= high:
                for label in [tick.label1, tick.label2]:
                    if label.get_visible() and label.get_text():
                        boxes.append(label.get_window_extent(renderer))
        offset = ax.yaxis.get_offset_text()
        if offset.get_visible() and offset.get_text():
            boxes.append(offset.get_window_extent(renderer))
    return left_spine, min((box.x0 for box in boxes), default=left_spine)


def place_shared_ylabel(fig, axes, text: str, *, fontsize: float = 8.0,
                        gap_pt: float = SHARED_YLABEL_GAP_PT,
                        outer_padding_pt: float = SHARED_YLABEL_OUTER_PADDING_PT):
    """Use a point-based gap from visible tick text and center on the data panels.

    Call after subplots_adjust. The left subplot margin is fitted to the label,
    tick text and outer padding while keeping all data limits and other subplot
    settings unchanged. Figure-coordinate constants are not used for spacing.
    """
    panels = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    if not panels or any(ax.figure is not fig for ax in panels):
        raise ValueError("Expected data panels from the supplied figure")
    if gap_pt <= 0 or outer_padding_pt <= 0:
        raise ValueError("Label gap and outer padding must be positive")
    ylabel = fig.supylabel(text, fontsize=fontsize, rotation=90,
                          ha="right", va="center")
    fig.canvas.draw()
    pixels_per_point = fig.dpi / 72
    left_spine, tick_left = _left_tick_boundary(fig, panels)
    label_width = ylabel.get_window_extent(fig.canvas.get_renderer()).width
    desired_left = (outer_padding_pt + gap_pt) * pixels_per_point + label_width + left_spine - tick_left
    left_fraction = desired_left / fig.bbox.width
    if not 0 < left_fraction < fig.subplotpars.right - .2:
        raise ValueError("Shared y label would leave insufficient plotting width")
    fig.subplots_adjust(left=left_fraction)
    fig.canvas.draw()
    _, tick_left = _left_tick_boundary(fig, panels)
    right_edge = tick_left - gap_pt * pixels_per_point
    vertical_center = (min(ax.bbox.y0 for ax in panels) + max(ax.bbox.y1 for ax in panels)) / 2
    ylabel.set_position((right_edge / fig.bbox.width, vertical_center / fig.bbox.height))
    fig.canvas.draw()
    box = ylabel.get_window_extent(fig.canvas.get_renderer())
    measured_gap = (tick_left - box.x1) / pixels_per_point
    measured_center_error = ((box.y0 + box.y1) / 2 - vertical_center) / pixels_per_point
    if abs(measured_gap - gap_pt) > .05 or abs(measured_center_error) > .05:
        raise ValueError("Shared y label spacing or centering differs from its specification")
    if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
        raise ValueError("Shared y label extends outside the figure canvas")
    return ylabel
