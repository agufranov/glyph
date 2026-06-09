#!/usr/bin/env python3
"""Extract glyph rectangles from TGA font atlases.

The script does not try to OCR the characters.  Instead it detects non-empty
rows and glyph bounds by the alpha channel, then prints a deterministic list of
rectangles.  If a charset string is supplied, detected rectangles are paired
with characters in reading order.

Examples:
  python3 extract_glyphs.py
  python3 extract_glyphs.py source/*.tga
  python3 extract_glyphs.py --chars "ABC..." fuente\ con\ sombra0000.tga
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


PREVIEW_BACKGROUND = (128, 96, 192, 255)


@dataclass(frozen=True)
class Glyph:
    file: str
    index: int
    line: int
    line_index: int
    x: int
    y: int
    width: int
    height: int
    baseline: int
    ink_x: int
    ink_y: int
    ink_width: int
    ink_height: int
    char: str | None = None


def make_mask(image: Image.Image, alpha_threshold: int) -> list[list[bool]]:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    width, height = rgba.size
    return [
        [alpha.getpixel((x, y)) > alpha_threshold for x in range(width)]
        for y in range(height)
    ]


def runs(values: Iterable[int | bool], min_gap: int = 1) -> list[tuple[int, int]]:
    """Return half-open ranges where values are non-zero.

    Gaps shorter than min_gap are bridged.  That helps with anti-aliased fonts
    where one transparent row/column can appear inside a single glyph/diacritic.
    """

    raw: list[tuple[int, int]] = []
    start: int | None = None
    items = list(values)

    for i, value in enumerate(items):
        if value and start is None:
            start = i
        elif not value and start is not None:
            raw.append((start, i))
            start = None
    if start is not None:
        raw.append((start, len(items)))

    if not raw or min_gap <= 1:
        return raw

    merged = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def trim_box(mask: list[list[bool]], x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if mask[y][x]:
                xs.append(x)
                ys.append(y)
    if not xs:
        return x0, y0, x1, y1
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def split_connected_components(
    mask: list[list[bool]], x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int, int, int]]:
    """Split touching glyphs into connected components inside one row.

    This is useful for rows where diacritics overlap neighbours horizontally:
    a pure column-projection splitter sees the whole row as one long run, while
    connected components still separate the visible ink islands.
    """

    seen: set[tuple[int, int]] = set()
    boxes: list[tuple[int, int, int, int]] = []

    for y in range(y0, y1):
        for x in range(x0, x1):
            if not mask[y][x] or (x, y) in seen:
                continue

            stack = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []

            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx in range(cx - 1, cx + 2):
                    for ny in range(cy - 1, cy + 2):
                        if (
                            nx < x0
                            or nx >= x1
                            or ny < y0
                            or ny >= y1
                            or (nx, ny) in seen
                            or not mask[ny][nx]
                        ):
                            continue
                        seen.add((nx, ny))
                        stack.append((nx, ny))

            boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))

    return sorted(boxes, key=lambda box: (box[1], box[0]))


def assign_components_to_lines(
    components: list[tuple[int, int, int, int]], min_row_gap: int
) -> list[list[tuple[int, int, int, int]]]:
    lines: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    current_bottom = 0

    for box in components:
        if not current or box[1] - current_bottom < min_row_gap:
            current.append(box)
            current_bottom = max(current_bottom, box[3])
        else:
            lines.append(sorted(current, key=lambda item: item[0]))
            current = [box]
            current_bottom = box[3]

    if current:
        lines.append(sorted(current, key=lambda item: item[0]))
    return lines


def merge_marks_with_bases(
    lines: list[list[tuple[int, int, int, int]]], max_mark_height: int
) -> list[list[tuple[int, int, int, int]]]:
    if max_mark_height <= 0:
        return lines

    merged_lines: list[list[tuple[int, int, int, int]]] = []
    for line in lines:
        result: list[tuple[int, int, int, int]] = []
        for box in sorted(line, key=lambda item: (item[1], item[0])):
            x0, y0, x1, y1 = box
            height = y1 - y0
            candidates = [
                index
                for index, current in enumerate(result)
                if height <= max_mark_height and min(x1, current[2]) > max(x0, current[0])
            ]
            if candidates:
                target_index = min(
                    candidates,
                    key=lambda index: abs(((result[index][0] + result[index][2]) / 2) - ((x0 + x1) / 2)),
                )
                cx0, cy0, cx1, cy1 = result[target_index]
                result[target_index] = (min(cx0, x0), min(cy0, y0), max(cx1, x1), max(cy1, y1))
            else:
                result.append(box)
        merged_lines.append(sorted(result, key=lambda item: item[0]))
    return merged_lines


def detect_lines_by_row_density(
    mask: list[list[bool]], width: int, height: int, threshold_ratio: float, min_row_gap: int
) -> list[tuple[int, int]]:
    row_projection = [sum(mask[y][x] for x in range(width)) for y in range(height)]
    row_threshold = max(1, int(max(row_projection, default=0) * threshold_ratio))
    dense_ranges = runs([value >= row_threshold for value in row_projection], min_gap=min_row_gap)

    line_ranges: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(dense_ranges):
        previous_end = dense_ranges[index - 1][1] if index else -1
        next_start = dense_ranges[index + 1][0] if index + 1 < len(dense_ranges) else height

        y0 = start
        while y0 > previous_end + 1 and row_projection[y0 - 1] > 0:
            y0 -= 1

        y1 = end
        while y1 < next_start - 1 and row_projection[y1] > 0:
            y1 += 1

        line_ranges.append((y0, y1))
    return line_ranges


def estimate_baseline(boxes: list[tuple[int, int, int, int]], line_top: int, line_height: int) -> int:
    if not boxes:
        return line_top + line_height - 1

    bottom_counts: dict[int, int] = {}
    for _x0, y0, _x1, y1 in boxes:
        height = y1 - y0
        if height < line_height * 0.35:
            continue
        bottom = y1 - 1
        bottom_counts[bottom] = bottom_counts.get(bottom, 0) + 1

    if not bottom_counts:
        return line_top + int(line_height * 0.8)
    return max(bottom_counts.items(), key=lambda item: (item[1], item[0]))[0]


def glyphs_from_boxes(
    path: Path,
    lines: list[list[tuple[int, int, int, int]]],
    line_ranges: list[tuple[int, int]] | None = None,
) -> list[Glyph]:
    glyphs: list[Glyph] = []
    for line_number, line in enumerate(lines):
        if line_ranges is None:
            line_top = min((box[1] for box in line), default=0)
            line_bottom = max((box[3] for box in line), default=line_top + 1)
        else:
            line_top, line_bottom = line_ranges[line_number]
        line_height = line_bottom - line_top
        baseline = estimate_baseline(line, line_top, line_height)

        for line_index, (x0, y0, x1, y1) in enumerate(line):
            glyphs.append(
                Glyph(
                    file=path.name,
                    index=len(glyphs),
                    line=line_number,
                    line_index=line_index,
                    x=x0,
                    y=line_top,
                    width=x1 - x0,
                    height=line_height,
                    baseline=baseline,
                    ink_x=x0,
                    ink_y=y0,
                    ink_width=x1 - x0,
                    ink_height=y1 - y0,
                )
            )
    return glyphs


def fixed_height_ranges(
    row_ranges: list[tuple[int, int]],
    image_height: int,
    force_step: int | None = None,
    force_first_top: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """Return uniformly-spaced, equal-height row ranges.

    If *force_step* / *force_first_top* are given they are used directly.
    Returns (ranges, step_used).
    """
    if len(row_ranges) < 2:
        return row_ranges, 0

    tops = [start for start, _end in row_ranges]

    if force_first_top is not None:
        first_top = force_first_top
    else:
        first_top = tops[0]

    if force_step is None:
        steps = [b - a for a, b in zip(tops, tops[1:])]
        force_step = sorted(steps)[len(steps) // 2]

    return [
        (first_top + i * force_step, min(image_height, first_top + (i + 1) * force_step))
        for i in range(len(row_ranges))
    ], force_step


def split_columns_by_body(
    mask: list[list[bool]],
    width: int,
    row_start: int,
    row_end: int,
    min_col_gap: int,
    body_top_ratio: float,
) -> list[tuple[int, int, int, int]]:
    full_projection = [sum(mask[y][x] for y in range(row_start, row_end)) for x in range(width)]
    full_ranges = runs(full_projection, min_gap=min_col_gap)
    boxes: list[tuple[int, int, int, int]] = []

    body_start = row_start + max(0, int((row_end - row_start) * body_top_ratio))
    for full_start, full_end in full_ranges:
        body_projection = [
            sum(mask[y][x] for y in range(body_start, row_end))
            for x in range(full_start, full_end)
        ]
        body_ranges = [(start + full_start, end + full_start) for start, end in runs(body_projection, min_gap=min_col_gap)]

        if len(body_ranges) <= 1:
            boxes.append(trim_box(mask, full_start, row_start, full_end, row_end))
            continue

        boundaries = [full_start]
        for left, right in zip(body_ranges, body_ranges[1:]):
            boundaries.append((left[1] + right[0]) // 2)
        boundaries.append(full_end)

        for left, right in zip(boundaries, boundaries[1:]):
            boxes.append(trim_box(mask, left, row_start, right, row_end))

    return boxes


def split_wide_boxes(
    boxes: list[tuple[int, int, int, int]], median_width_factor: float
) -> list[tuple[int, int, int, int]]:
    widths = sorted(x1 - x0 for x0, _y0, x1, _y1 in boxes if x1 > x0)
    if not widths:
        return boxes

    median_width = widths[len(widths) // 2]
    max_width = max(1, int(median_width * median_width_factor))

    result: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        width = x1 - x0
        if width <= max_width:
            result.append((x0, y0, x1, y1))
            continue

        parts = max(2, round(width / median_width))
        step = width / parts
        for index in range(parts):
            left = round(x0 + index * step)
            right = round(x0 + (index + 1) * step)
            result.append((left, y0, right, y1))

    return result


def extract_from_image(
    path: Path,
    alpha_threshold: int,
    min_row_gap: int,
    min_col_gap: int,
    method: str,
    row_density_threshold: float,
    max_mark_height: int,
    body_top_ratio: float,
    split_wide_factor: float,
) -> list[Glyph]:
    image = Image.open(path).convert("RGBA")
    width, height = image.size
    mask = make_mask(image, alpha_threshold)

    if method == "components":
        components = split_connected_components(mask, 0, 0, width, height)
        lines = assign_components_to_lines(components, min_row_gap)
        return glyphs_from_boxes(path, merge_marks_with_bases(lines, max_mark_height))

    if method == "rows-columns":
        row_ranges = detect_lines_by_row_density(mask, width, height, row_density_threshold, min_row_gap)
    else:
        row_projection = [sum(mask[y][x] for x in range(width)) for y in range(height)]
        row_ranges = runs(row_projection, min_gap=min_row_gap)

    line_ranges, _step = fixed_height_ranges(row_ranges, height)
    lines = []
    for row_start, row_end in line_ranges:
        boxes = split_columns_by_body(mask, width, row_start, row_end, min_col_gap, body_top_ratio)
        lines.append(split_wide_boxes(boxes, split_wide_factor))
    return glyphs_from_boxes(path, lines, line_ranges)


def extract_from_clean_image(
    path: Path,
    shadow_threshold: int,
    min_row_gap: int,
    force_offset: int | None = None,
    force_step: int | None = None,
    force_first_top: int | None = None,
) -> tuple[list[Glyph], int, int, int]:
    image = remove_shadow(Image.open(path), shadow_threshold)
    width, height = image.size
    alpha = image.getchannel("A")
    mask = [[alpha.getpixel((x, y)) > 0 for x in range(width)] for y in range(height)]

    row_ranges = detect_lines_by_row_density(mask, width, height, 0.35, min_row_gap)
    if not row_ranges:
        return [], 0, 0

    # Build per-row boxes; drop completely empty rows.
    all_lines: list[list[tuple[int, int, int, int]]] = []
    kept_ranges: list[tuple[int, int]] = []
    for row_start, row_end in row_ranges:
        col_projection = [
            sum(alpha.getpixel((x, y)) > 0 for y in range(row_start, row_end))
            for x in range(width)
        ]
        line: list[tuple[int, int, int, int]] = []
        for col_start, col_end in runs(col_projection):
            ys = [
                y
                for y in range(row_start, row_end)
                for x in range(col_start, col_end)
                if alpha.getpixel((x, y)) > 0
            ]
            if ys:
                line.append((col_start, min(ys), col_end, max(ys) + 1))
        if line:
            all_lines.append(line)
            kept_ranges.append((row_start, row_end))

    if not all_lines:
        return [], 0, 0

    kept_ranges, actual_step = fixed_height_ranges(kept_ranges, height, force_step, force_first_top)

    glyphs, offset = normalize_baselines(
        glyphs_from_boxes(path, all_lines, kept_ranges), kept_ranges, force_offset
    )
    first_top = kept_ranges[0][0] if kept_ranges else 0
    return glyphs, offset, actual_step, first_top


def normalize_baselines(
    glyphs: list[Glyph],
    line_ranges: list[tuple[int, int]],
    force_offset: int | None = None,
) -> tuple[list[Glyph], int]:
    """Apply a uniform baseline-offset to every non-empty row.

    If *force_offset* is given it is used directly.  Otherwise the offset is
    computed from the most frequent glyph bottom in rows 1 and 2 (which
    contain capital letters sitting squarely on the baseline).

    Returns (glyphs, offset_used).
    """
    zero = glyphs, 0
    if not line_ranges or not glyphs:
        return zero

    by_line: dict[int, list[Glyph]] = {}
    for g in glyphs:
        by_line.setdefault(g.line, []).append(g)

    if force_offset is None:
        # Only consider rows 1 and 2 for the baseline reference
        reference_lines = [1, 2]
        available = [li for li in reference_lines if li < len(line_ranges) and li in by_line]
        if not available:
            return zero

        all_bottoms: list[int] = []
        for line_idx in available:
            top, bottom = line_ranges[line_idx]
            line_height = bottom - top
            for g in by_line[line_idx]:
                h = g.ink_y + g.ink_height - g.y
                if h < line_height * 0.35:
                    continue
                all_bottoms.append(g.ink_y + g.ink_height - 1)

        if not all_bottoms:
            return zero

        # Descender filter: keep bottoms that are at most 2 px below median
        sorted_bottoms = sorted(all_bottoms)
        median_bottom = sorted_bottoms[len(sorted_bottoms) // 2]
        filtered = [b for b in all_bottoms if b <= median_bottom + 2]
        if not filtered:
            filtered = all_bottoms

        # Mode of filtered bottoms
        counts: dict[int, int] = {}
        for b in filtered:
            counts[b] = counts.get(b, 0) + 1
        reference_baseline = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

        force_offset = reference_baseline - line_ranges[available[0]][0]

    return [
        replace(g, baseline=line_ranges[g.line][0] + force_offset)
        for g in glyphs
    ], force_offset


def assign_chars(glyphs: list[Glyph], chars: str | None) -> list[Glyph]:
    if chars is None:
        return glyphs
    assigned: list[Glyph] = []
    for glyph, char in zip(glyphs, chars):
        assigned.append(replace(glyph, char=char))
    assigned.extend(glyphs[len(chars) :])
    return assigned


def save_debug_image(path: Path, glyphs: list[Glyph], output: Path, shadow_threshold: int) -> None:
    image = remove_shadow(Image.open(path), shadow_threshold)
    background = Image.new("RGBA", image.size, PREVIEW_BACKGROUND)
    background.alpha_composite(image)

    draw = ImageDraw.Draw(background)
    for glyph in glyphs:
        left = glyph.ink_x
        right = glyph.ink_x + glyph.ink_width - 1
        draw.line((left, glyph.y, left, glyph.y + glyph.height - 1), fill=(255, 60, 60, 255))
        draw.line((right, glyph.y, right, glyph.y + glyph.height - 1), fill=(255, 60, 60, 255))
        draw.line((left, glyph.baseline, right, glyph.baseline), fill=(80, 255, 120, 255))
        draw.text((left, max(0, glyph.y - 8)), str(glyph.index), fill=(80, 220, 255, 255))

    background.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(output)


def save_plain_image(path: Path, output: Path) -> None:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, PREVIEW_BACKGROUND)
    background.alpha_composite(image)
    background.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(output)


def is_shadow_pixel(r: int, g: int, b: int, a: int, threshold: int) -> bool:
    return a > 0 and r <= threshold and g <= threshold and b <= threshold


def is_foreground_pixel(r: int, g: int, b: int, a: int, threshold: int) -> bool:
    return a > 0 and not is_shadow_pixel(r, g, b, a, threshold)


def estimate_shadow_params(image: Image.Image, shadow_threshold: int) -> dict[str, object]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    pixels = rgba.load()

    foreground: set[tuple[int, int]] = set()
    shadow: set[tuple[int, int]] = set()
    shadow_alpha: list[int] = []
    shadow_rgb: dict[tuple[int, int, int], int] = {}

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if is_shadow_pixel(r, g, b, a, shadow_threshold):
                shadow.add((x, y))
                shadow_alpha.append(a)
                shadow_rgb[(r, g, b)] = shadow_rgb.get((r, g, b), 0) + 1
            elif is_foreground_pixel(r, g, b, a, shadow_threshold):
                foreground.add((x, y))

    best_offset = (0, 0)
    best_score = -1
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            score = sum((x + dx, y + dy) in shadow for x, y in foreground)
            if score > best_score:
                best_score = score
                best_offset = (dx, dy)

    shifted_foreground = {(x + best_offset[0], y + best_offset[1]) for x, y in foreground}
    max_distance = 0
    for sx, sy in shadow:
        if (sx, sy) in shifted_foreground:
            continue
        for radius in range(1, 9):
            found = False
            for yy in range(sy - radius, sy + radius + 1):
                for xx in range(sx - radius, sx + radius + 1):
                    if (xx, yy) in shifted_foreground:
                        found = True
                        break
                if found:
                    break
            if found:
                max_distance = max(max_distance, radius)
                break

    color = max(shadow_rgb.items(), key=lambda item: item[1])[0] if shadow_rgb else (0, 0, 0)
    return {
        "offset": {"x": best_offset[0], "y": best_offset[1]},
        "blur_radius_estimate": max_distance,
        "color_rgb": list(color),
        "alpha_min": min(shadow_alpha) if shadow_alpha else 0,
        "alpha_max": max(shadow_alpha) if shadow_alpha else 0,
        "shadow_pixels": len(shadow),
        "foreground_pixels": len(foreground),
        "shadow_threshold": shadow_threshold,
        "note": "Estimated from black/near-black pixels; original atlas stores shadow as black RGBA pixels with varying alpha.",
    }


def remove_shadow(image: Image.Image, shadow_threshold: int) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if is_shadow_pixel(r, g, b, a, shadow_threshold):
                pixels[x, y] = (0, 0, 0, 0)
                continue

            if a > 0:
                # The original atlas already has white glyphs composited over a
                # black shadow.  On anti-aliased edges this produces gray pixels:
                # straight_rgb ~= white_alpha / combined_alpha.  Recover the
                # glyph alpha from the premultiplied luminance and make the glyph
                # white again, so no gray shadow fringe remains.
                recovered_alpha = round(a * max(r, g, b) / 255)
                pixels[x, y] = (255, 255, 255, recovered_alpha) if recovered_alpha else (0, 0, 0, 0)
    return rgba


def save_no_shadow_image(path: Path, output: Path, shadow_threshold: int) -> None:
    image = remove_shadow(Image.open(path), shadow_threshold)
    background = Image.new("RGBA", image.size, PREVIEW_BACKGROUND)
    background.alpha_composite(image)
    background.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(output)


def save_glyph_images(path: Path, glyphs: list[Glyph], output_dir: Path) -> None:
    image = Image.open(path).convert("RGBA")
    output_dir.mkdir(parents=True, exist_ok=True)
    for glyph in glyphs:
        crop = image.crop((glyph.x, glyph.y, glyph.x + glyph.width, glyph.y + glyph.height))
        name_char = f"_{ord(glyph.char):04x}" if glyph.char else ""
        crop.save(output_dir / f"{glyph.index:03d}{name_char}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect glyph bounds in TGA font atlases.")
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="TGA/PNG images to process; defaults to source/*.tga",
    )
    parser.add_argument("--alpha-threshold", type=int, default=0, help="minimum alpha considered visible")
    parser.add_argument("--min-row-gap", type=int, default=2, help="bridge vertical gaps smaller than this")
    parser.add_argument("--min-col-gap", type=int, default=1, help="bridge horizontal gaps smaller than this")
    parser.add_argument(
        "--method",
        choices=("rows-columns", "columns", "components"),
        default="rows-columns",
        help="glyph splitting method; rows-columns is best for the supplied atlases",
    )
    parser.add_argument(
        "--row-density-threshold",
        type=float,
        default=0.35,
        help="relative row-density threshold used by rows-columns line detection",
    )
    parser.add_argument(
        "--max-mark-height",
        type=int,
        default=8,
        help="components mode: merge small accents/punctuation marks into overlapping base glyphs",
    )
    parser.add_argument(
        "--body-top-ratio",
        type=float,
        default=0.3,
        help="ignore the top part of a row while looking for column gaps; helps split ij/iiii",
    )
    parser.add_argument(
        "--split-wide-factor",
        type=float,
        default=1.8,
        help="split boxes wider than median_width * factor; helps with glued ij/iiii runs",
    )
    parser.add_argument("--chars", help="characters in atlas order; optional, no OCR is performed")
    parser.add_argument("--out-dir", type=Path, default=Path("result"), help="directory for generated files")
    parser.add_argument("--json", type=Path, help="manifest output path; defaults to result/glyph_manifest.json")
    parser.add_argument("--no-debug", action="store_true", help="do not write *_debug.png with rectangles")
    parser.add_argument("--no-plain", action="store_true", help="do not write *_plain.png on green background")
    parser.add_argument("--no-no-shadow", action="store_true", help="do not write *_no_shadow.png")
    parser.add_argument("--shadow-threshold", type=int, default=16, help="RGB threshold for black shadow detection")
    parser.add_argument("--no-save-glyphs", action="store_true", help="do not save each glyph as a separate PNG")
    args = parser.parse_args()

    image_paths = args.images or sorted(Path("source").glob("*.tga"))
    if not image_paths:
        parser.error("no input images found; pass images explicitly or put .tga files into source/")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.json or args.out_dir / "glyph_manifest.json"

    all_glyphs: list[Glyph] = []
    shadow_params_by_file: dict[str, object] = {}
    baseline_offset: int | None = None
    baseline_step: int | None = None
    baseline_first_top: int | None = None
    for image_path in image_paths:
        source_image = Image.open(image_path).convert("RGBA")
        shadow_params_by_file[image_path.name] = estimate_shadow_params(source_image, args.shadow_threshold)
        glyphs, used_offset, used_step, used_first_top = extract_from_clean_image(
            image_path, args.shadow_threshold, args.min_row_gap,
            baseline_offset, baseline_step, baseline_first_top,
        )
        if baseline_offset is None:
            baseline_offset = used_offset
        if baseline_step is None:
            baseline_step = used_step
        if baseline_first_top is None:
            baseline_first_top = used_first_top
        glyphs = assign_chars(glyphs, args.chars if len(args.images) == 1 else None)
        all_glyphs.extend(glyphs)

        print(f"\n{image_path.name}: {len(glyphs)} glyphs")
        for glyph in glyphs:
            label = f" {glyph.char!r}" if glyph.char is not None else ""
            print(
                f"#{glyph.index:03d}{label} line={glyph.line:02d} "
                f"x={glyph.x:3d} y={glyph.y:3d} w={glyph.width:2d} h={glyph.height:2d} "
                f"baseline={glyph.baseline:3d} "
                f"ink=({glyph.ink_x},{glyph.ink_y},{glyph.ink_width},{glyph.ink_height})"
            )

        if not args.no_debug:
            save_debug_image(image_path, glyphs, args.out_dir / f"{image_path.stem}_debug.png", args.shadow_threshold)
        if not args.no_plain:
            save_plain_image(image_path, args.out_dir / f"{image_path.stem}_plain.png")
        if not args.no_no_shadow:
            save_no_shadow_image(image_path, args.out_dir / f"{image_path.stem}_no_shadow.png", args.shadow_threshold)
        if not args.no_save_glyphs:
            save_glyph_images(image_path, glyphs, args.out_dir / "glyphs" / image_path.stem)

    (args.out_dir / "shadow_params.json").write_text(
        json.dumps(shadow_params_by_file, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps([asdict(glyph) for glyph in all_glyphs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Shadow params written to {args.out_dir / 'shadow_params.json'}")
    print(f"\nManifest written to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
