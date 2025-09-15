import subprocess
from pathlib import Path
from svgelements import SVG, Group, Path as SVGPath, Shape, Matrix, Title, Desc

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

def remove_unwanted_elements(element, remove_classes=("Page", "Tempo")):
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
            apply_offset(list(element.elements())[:1], offset)

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
            apply_offset(elem, offset)
            merged.append(elem)
        offset += svg.height

    print("done.")
    return merged

def vertical_bbox_distance(a: Shape, b: Shape) -> float:
    """Return the minimum Euclidean distance between two shapes' bounding boxes.
    If boxes overlap, distance is 0. If either bbox is None return a large number.
    """
    ba = a.bbox()
    bb = b.bbox()
    if ba is None or bb is None:
        return float("inf")
    ax1, ay1, ax2, ay2 = ba[0], ba[1], ba[2], ba[3]
    bx1, by1, bx2, by2 = bb[0], bb[1], bb[2], bb[3]

    if ay2 < by1:
        return by1 - ay2
    elif by2 < ay1:
        return ay1 - by2
    else:
        return 0.0

def group_by_brackets(svg: SVG):
    """
    Group elements by bracket alignment using bounding boxes.
    """
    print("Grouping merged SVG by brackets...", end="")
    brackets = []

    for elem in list(svg.elements())[1:]:
        if "Bracket" in getattr(elem, "values", {}).get("class", ""):
            top, bottom = find_top_and_bottom(elem)
            brackets.append({"elem": elem, "top": top, "bottom": bottom, "members": []})
    brackets.sort(key=lambda b: b["top"])

    unassigned = []

    for elem in list(svg.elements())[1:]:
        if elem in (b["elem"] for b in brackets):
            continue
        top, bottom = find_top_and_bottom(elem)
        assigned = False
        for b in brackets:
            if top >= b["top"] and bottom <= b["bottom"]:
                b["members"].append(elem)
                assigned = True
                break
        if not assigned:
            unassigned.append(elem)

    if brackets and unassigned:
        for elem in unassigned:
            best = None
            best_dist = float("inf")
            for b in brackets:
                d = vertical_bbox_distance(elem, b["elem"])
                if d < best_dist:
                    best_dist = d
                    best = b
            if best is not None:
                best["members"].append(elem)

    grouped_svg = SVG()
    grouped_svg.width = svg.width
    grouped_svg.height = svg.height

    for i, b in enumerate(brackets):
        g = Group(id=f"Group-{i}")
        g.append(b["elem"])
        sorted_members = sorted(b["members"], key=find_x_coordinate)
        for m in sorted_members:
            g.append(m)
        grouped_svg.append(g)

    print("done.")
    return grouped_svg


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
    grouped = group_by_brackets(merged)

    grouped_out = out_dir / (midi_path.stem + ".svg")
    grouped.write_xml(str(grouped_out))

    for p in fixed_paths:
        if p != grouped_out and p.exists():
            p.unlink()

    print(f"Done. Final merged SVG: {grouped_out}")


if __name__ == "__main__":
    midi_to_svg(
        "/home/u200b/Music/Credits Song For My Death.mid",
        "./test/",
        "mscore"
    )
