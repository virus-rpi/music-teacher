import subprocess
from pathlib import Path
from svgelements import SVG, Group, Path as SVGPath, Shape, Matrix, Title, Desc
import bisect
import copy

def run_musescore(midi_path: Path, svg_path: Path, mscore_cmd="mscore"):
    """
    Call MuseScore to export MIDI to SVG.
    """
    cmd = [mscore_cmd, str(midi_path), "-o", str(svg_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def fix_svg_scaling(svg_in: Path, svg_out: Path):
    """
    Parse an SVG and normalize MuseScore's tiny scale transforms.
    """
    svg = SVG().parse(str(svg_in))
    scale_factor = 65536.0
    for element in list(svg.elements())[1:]:
        if hasattr(element, "transform") and element.transform is not None:
            element.transform *= f"scale({scale_factor})"
    svg.write_xml(str(svg_out))

def find_top_and_bottom(element: Shape) -> tuple[float, float]:
    """
    Use svgelements bounding box to get top and bottom of an element.
    """
    bbox = element.bbox()
    if bbox is None:
        return 0.0, 0.0
    return bbox[1], bbox[3]

def find_x_coordinate(element: Shape) -> float:
    """
    Use svgelements bounding box to get left x coordinate of an element.
    """
    bbox = element.bbox()
    if bbox is None:
        return 0.0
    return (bbox[0] + bbox[2]) / 2.0

def looks_like_background(element: Shape) -> bool:
    """
    Detect MuseScore background rectangles/paths.
    """
    if not hasattr(element, "fill") or element.fill is None:
        return False
    if str(element.fill).lower() not in ("#fff", "#ffffff", "white"):
        return False
    if isinstance(element, SVGPath):
        return element.d().startswith("M0,0") or element.d().startswith("M0 0") or element.d().startswith("M 0,0") or element.d().startswith("M 0 0")
    return False

def remove_unwanted_elements(element, remove_classes=("Page", "Tempo", "MeasureNumber")):
    if isinstance(element, list):
        new_elements = []
        for e in element:
            if not remove_unwanted_elements(e, remove_classes):
                new_elements.append(e)
        element[:] = new_elements
        return False
    elif isinstance(element, Group):
        new_elements = []
        for e in element.elements:
            if not remove_unwanted_elements(e, remove_classes):
                new_elements.append(e)
        element.elements = new_elements
    return "class" in element.values and any(c in element.values["class"] for c in remove_classes) or isinstance(element, (Title, Desc)) or looks_like_background(element)

def apply_offset(element, offset):
    """Recursively apply vertical translation to element(s)."""
    if isinstance(element, list):
        for e in element:
            apply_offset(e, offset)
    else:
        element *= Matrix.translate_y(offset)
        if hasattr(element, 'elements'):
            apply_offset(list(element.elements()), offset)

def merge_svgs_to_long_page(svg_paths):
    """
    Merge multiple SVGs vertically into one long SVG.
    """
    print("Merging SVGs to long page...", end="")
    svgs = [SVG().parse(str(p)) for p in svg_paths]

    for svg in svgs:
        remove_unwanted_elements(svg)

    total_height = sum(svg.height for svg in svgs)
    max_width = max(svg.width for svg in svgs)

    merged = SVG()
    merged.width = max_width
    merged.height = total_height

    offset = 0
    for svg in svgs:
        for elem in list(svg.elements())[1:]:
            elem_copy = copy.deepcopy(elem)
            apply_offset(elem_copy, offset)
            merged.append(elem_copy)
        offset += svg.height

    print("done.")
    return merged

def group_by_bracket(svg: SVG):
    """
    Group elements by bracket alignment using bounding boxes.
    """
    print("Grouping merged SVG by brackets...", end="")

    elements = list(svg.elements())[1:]
    if not elements:
        print("done.")
        return svg

    cached = []
    for elem in elements:
        bbox = elem.bbox()
        if bbox is None:
            left = right = top = bottom = 0.0
        else:
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        center = (top + bottom) / 2.0
        x_center = (left + right) / 2.0
        cached.append({
            "elem": elem,
            "bbox": bbox,
            "top": top,
            "bottom": bottom,
            "center": center,
            "x": x_center,
        })

    brackets = [c.copy() for c in cached if "Bracket" in getattr(c["elem"], "values", {}).get("class", "")]

    if not brackets:
        print("done.")
        return svg

    brackets.sort(key=lambda b: b["center"])  # in-place
    bracket_centers = [b["center"] for b in brackets]
    for b in brackets:
        b["members"] = []
        b["members_x"] = []

    bracket_elems = [b["elem"] for b in brackets]

    def vdist(ba_bbox, bb_bbox):
        if ba_bbox is None or bb_bbox is None:
            return float("inf")
        ax1, ay1, ax2, ay2 = ba_bbox[0], ba_bbox[1], ba_bbox[2], ba_bbox[3]
        bx1, by1, bx2, by2 = bb_bbox[0], bb_bbox[1], bb_bbox[2], bb_bbox[3]
        if ay2 < by1:
            return by1 - ay2
        elif by2 < ay1:
            return ay1 - by2
        else:
            return 0.0

    unassigned = []
    for c in cached:
        elem = c["elem"]
        if elem in bracket_elems:
            continue
        top = c["top"]
        bottom = c["bottom"]
        bbox = c["bbox"]
        center = c["center"]
        x = c["x"]
        assigned = False
        for b in brackets:
            if top >= b["top"] and bottom <= b["bottom"]:
                b["members"].append(elem)
                b["members_x"].append(x)
                assigned = True
                break
        if assigned:
            continue
        idx = bisect.bisect_left(bracket_centers, center)
        best = None
        best_dist = float("inf")
        left_idx = max(0, idx - 3)
        right_idx = min(len(brackets), idx + 4)
        for j in range(left_idx, right_idx):
            b = brackets[j]
            d = vdist(bbox, b["bbox"])
            if d < best_dist:
                best_dist = d
                best = b
        if best is not None:
            best["members"].append(elem)
            best["members_x"].append(x)
        else:
            unassigned.append((elem, x))

    grouped_svg = SVG()
    grouped_svg.width = svg.width
    grouped_svg.height = svg.height

    for i, b in enumerate(brackets):
        g = Group(id=f"Group-{i}")
        g.append(copy.deepcopy(b["elem"]))
        if b["members"]:
            members_with_x = list(zip(b["members"], b["members_x"]))
            members_with_x.sort(key=lambda mx: mx[1])
            for m, _x in members_with_x:
                g.append(copy.deepcopy(m))
        grouped_svg.append(g)
    print("done.")
    return grouped_svg

def group_by_measure(svg: SVG) -> SVG:
    """
    Split each bracket group into measures based on BarLine elements.
    Each measure becomes a <Group id="...-Measure-n"> with:
      - StaffLines inside their own <Group>
      - Other elements grouped normally, sorted by x
    """
    print("Grouping by measures...", end="")

    new_svg = SVG()
    new_svg.width = svg.width
    new_svg.height = svg.height

    for group in [g for g in list(svg.elements())[1:] if isinstance(g, Group)]:
        elems = list(group)

        bar_lines = [e for e in elems if "BarLine" in getattr(e, "values", {}).get("class", "")]
        bar_infos = []
        for bl in bar_lines:
            bb = bl.bbox()
            if not bb:
                continue
            x = (bb[0] + bb[2]) / 2
            bar_infos.append((x, bl))
        bar_infos.sort(key=lambda b: b[0])

        bar_positions = []
        current_x = None
        cluster = []
        for x, bl in bar_infos:
            if current_x is None or abs(x - current_x) < 1.0:
                cluster.append(bl)
                current_x = x
            else:
                bar_positions.append((current_x, cluster))
                cluster = [bl]
                current_x = x
        if cluster:
            bar_positions.append((current_x, cluster))

        measures = []
        scaffolding = Group(id=f"{group.values.get('id')}-Scaffolding")
        for i in range(len(bar_positions) - 1):
            x_start = bar_positions[i][0]
            x_end = bar_positions[i+1][0]

            content = []

            for e in elems:
                bb = e.bbox()
                if not bb:
                    continue
                ex = (bb[0] + bb[2]) / 2
                if x_start <= ex <= x_end:
                    if "StaffLine" in getattr(e, "values", {}).get("class", ""):
                        scaffolding.append(e)
                    elif "BarLine" in getattr(e, "values", {}).get("class", ""):
                        scaffolding.append(e)
                    else:
                        content.append(e)

            content.sort(key=lambda e: (e.bbox()[0] if e.bbox() else 0))

            measure_group = Group(id=f"{group.values.get('id')}-Measure-{i}")
            for c in content:
                measure_group.append(c)
            measures.append(measure_group)
        new_group = Group(id=group.values.get("id"))
        new_group.append(scaffolding)
        for m in measures:
            new_group.append(m)
        new_svg.append(new_group)

    print("done.")
    return new_svg

def strip_duplicate_clefs(svg: SVG) -> SVG:
    print("Stripping duplicate clefs...", end="")

    # TODO

    print("done.")
    return svg

def bracket_groups_to_long_strip(svg: SVG) -> SVG:
    print("Laying out bracket groups into one long strip...", end="")

    groups = [g for g in list(svg.elements())[0] if isinstance(g, Group)]
    if not groups:
        print("done.")
        return svg

    def compute_bbox(elem):
        bb = None
        # if this element has a bbox, start with it
        try:
            own = elem.bbox()
        except Exception:
            own = None
        if own:
            bb = [own[0], own[1], own[2], own[3]]
        # recurse into children if any
        if hasattr(elem, 'elements'):
            for child in list(elem.elements()):
                cb = compute_bbox(child)
                if cb:
                    if bb is None:
                        bb = [cb[0], cb[1], cb[2], cb[3]]
                    else:
                        bb[0] = min(bb[0], cb[0])
                        bb[1] = min(bb[1], cb[1])
                        bb[2] = max(bb[2], cb[2])
                        bb[3] = max(bb[3], cb[3])
        return tuple(bb) if bb is not None else None

    group_copies = [copy.deepcopy(g) for g in groups]

    bboxes = []
    widths = []
    heights = []
    for g in group_copies:
        bb = compute_bbox(g) or (0.0, 0.0, 0.0, 0.0)
        minx, miny, maxx, maxy = bb
        width = maxx - minx
        height = maxy - miny
        bboxes.append(bb)
        widths.append(width)
        heights.append(height)

    total_width = sum(widths)
    max_height = max(heights) if heights else 0.0

    out = SVG()
    out.width = total_width
    out.height = max_height

    current_x = 0.0
    for g_copy, bb, w in zip(group_copies, bboxes, widths):
        minx, miny, maxx, maxy = bb
        dx = current_x - minx
        dy = -miny

        g_copy *= Matrix.translate(dx, dy)
        for bar_line in [g for g in g_copy[0] if g.values["class"] == "BarLine"][2:-2]:
            bar_line *= Matrix.translate(-dx/2, -dy/2)
        out.append(g_copy)
        current_x += w

    print("done.")
    return out

def midi_to_svg(midi_file: str, out_dir: str, mscore_cmd="mscore"):
    midi_path = Path(midi_file).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in out_dir.glob("*.svg"):
        f.unlink()

    raw_svg = out_dir / (midi_path.stem + ".svg")
    run_musescore(midi_path, raw_svg, mscore_cmd=mscore_cmd)

    svg_paths = sorted(out_dir.glob(f"{midi_path.stem}*.svg"))
    if not svg_paths:
        raise FileNotFoundError(f"No SVG files generated for {midi_path.stem}")

    fixed_paths = []
    for svg in svg_paths:
        fixed = out_dir / svg.name
        fix_svg_scaling(svg, fixed)
        fixed_paths.append(fixed)

    merged = merge_svgs_to_long_page(fixed_paths)
    grouped = group_by_bracket(merged)
    grouped = group_by_measure(grouped)
    stripped_clefs = strip_duplicate_clefs(grouped)
    grouped = bracket_groups_to_long_strip(stripped_clefs)

    grouped_out = out_dir / (midi_path.stem + ".svg")
    grouped.write_xml(str(grouped_out))

    note_xs: list[float] = []
    try:
        for elem in grouped.elements():
            candidates = [elem]
            if hasattr(elem, 'elements'):
                candidates = list(elem.elements())
            for e in candidates:
                try:
                    vals = getattr(e, 'values', {}) or {}
                    cls = vals.get('class', '')
                except Exception:
                    cls = ''
                if str(cls) == "Note":
                    try:
                        bb = e.bbox()
                        if bb:
                            x = (bb[0] + bb[2]) / 2.0
                            note_xs.append(float(x))
                        else:
                            note_xs.append(float(find_x_coordinate(e)))
                    except Exception:
                        try:
                            note_xs.append(float(find_x_coordinate(e)))
                        except Exception:
                            continue
    except Exception:
        note_xs = []

    for p in fixed_paths:
        if p != grouped_out and p.exists():
            p.unlink()

    print(f"Done. Final merged SVG: {grouped_out}")
    return list(set(note_xs))

if __name__ == "__main__":
    midi_to_svg(
        "/home/u200b/Music/Credits Song For My Death.mid",
        "./.sheet_music_cache/",
        "mscore"
    )
