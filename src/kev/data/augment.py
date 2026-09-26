"""Quarter turns and mirroring of a view, with its boxes kept in step."""

from PIL import Image

Box = list[float] | None


def _turn_box(box: list[float], width: int) -> list[float]:
    # PIL's ROTATE_90 turns anticlockwise: (x, y) in a `width`-wide image lands at (y, width - x).
    x, y, w, h = box
    return [y, width - x - w, h, w]


def _mirror_box(box: list[float], width: int) -> list[float]:
    x, y, w, h = box
    return [width - x - w, y, w, h]


def orient(image: Image.Image, boxes: list[Box], quarter_turns: int, mirror: bool) -> tuple[Image.Image, list[Box]]:
    for _ in range(quarter_turns % 4):
        boxes = [_turn_box(b, image.width) if b else None for b in boxes]
        image = image.transpose(Image.Transpose.ROTATE_90)
    if mirror:
        boxes = [_mirror_box(b, image.width) if b else None for b in boxes]
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image, boxes
