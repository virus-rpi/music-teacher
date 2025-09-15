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

    print(f"Applying scale factor ≈ {scale_factor:.1f}")
    for element in list(svg.elements())[1:]:
        if hasattr(element, "transform") and element.transform is not None:
            element.transform *= f"scale({scale_factor})"

    svg.write_xml(str(svg_out))


def find_top_and_bottom(element: Shape) -> tuple[float, float]:
    """
    Use svgelements' bounding box to get top and bottom of an element.
    """
    bbox = element.bbox()
    if bbox is None:
        return 0.0, 0.0
    return bbox[1], bbox[3]


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


def group_merged_svg_by_brackets(svg: SVG):
    """
    Group elements by bracket alignment using bounding boxes.
    """
    print("Grouping merged SVG by brackets...", end="")
    groups = []
    brackets = []

    # find brackets
    for elem in list(svg.elements())[1:]:
        if "Bracket" in getattr(elem, "values", {}).get("class", ""):
            top, bottom = find_top_and_bottom(elem)
            brackets.append({"elem": elem, "top": top, "bottom": bottom, "members": []})

    # assign elements
    for elem in list(svg.elements())[1:]:
        if elem in (b["elem"] for b in brackets):
            continue
        top, bottom = find_top_and_bottom(elem)
        center = (top + bottom) / 2.0
        assigned = False
        for b in brackets:
            if b["top"] <= center <= b["bottom"]:
                b["members"].append(elem)
                assigned = True
                break
        if not assigned:
            groups.append(elem)

    grouped_svg = SVG()
    grouped_svg.width = svg.width
    grouped_svg.height = svg.height

    for b in brackets:
        g = Group(id="Group-Bracket")
        g.append(b["elem"])
        for m in b["members"]:
            g.append(m)
        grouped_svg.append(g)

    for e in groups:
        grouped_svg.append(e)

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
    grouped = group_merged_svg_by_brackets(merged)

    grouped_out = out_dir / (midi_path.stem + ".svg")
    merged_out = out_dir / (midi_path.stem + "_merged.svg")
    merged.write_xml(str(merged_out))
    grouped.write_xml(str(grouped_out))

    for p in fixed_paths:
        if p != merged_out and p.exists():
            p.unlink()

    print(f"Done. Final merged SVG: {merged_out}")


if __name__ == "__main__":
    midi_to_svg(
        "/home/u200b/Music/Credits Song For My Death.mid",
        "./test/",
        "mscore"
    )
