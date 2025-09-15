import subprocess
import re
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np

def run_musescore(midi_path: Path, svg_path: Path, mscore_cmd="mscore"):
    """
    Call MuseScore to export MIDI to SVG.
    """
    cmd = [mscore_cmd, str(midi_path), "-o", str(svg_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def fix_svg_scaling(svg_in: Path, svg_out: Path):
    """
    Parse an SVG and rescale transforms so drawings are visible.
    """
    with open(svg_in, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")

    scale_factor = None

    for tag in soup.find_all(attrs={"transform": True}):
        m = re.match(r"matrix\(([^)]+)\)", tag["transform"])
        if not m:
            continue
        nums = [float(x) for x in m.group(1).split(",")]

        # MuseScore bug: values like 1.52588e-5
        if scale_factor is None and (nums[0] < 0.01 or nums[3] < 0.01):
            # Inverse of the tiny scale
            scale_factor = 1.0 / nums[0] if nums[0] != 0 else 1.0 / nums[3]

        if scale_factor:
            nums[0] *= scale_factor
            nums[3] *= scale_factor
            nums[4] *= scale_factor
            nums[5] *= scale_factor
            tag["transform"] = "matrix({})".format(",".join(f"{x:.6f}" for x in nums))

    if scale_factor:
        print(f"Applied scale factor ≈ {scale_factor:.1f}")

    with open(svg_out, "w", encoding="utf-8") as f:
        f.write(str(soup))

def parse_transform_y(transform_str):
    if not transform_str:
        return 0.0
    m = re.search(r'matrix\([^)]*\)', transform_str)
    if m:
        nums = [float(x) for x in m.group(0).replace('matrix(', '').replace(')', '').split(',')]
        if len(nums) >= 6:
            return nums[5]
    m = re.search(r'translate\(([^)]+)\)', transform_str)
    if m:
        parts = [float(x) for x in re.split(r'[ ,]+', m.group(1).strip()) if x != '']
        if len(parts) == 1:
            return 0.0
        return parts[1]
    return 0.0

_transform_re = re.compile(
    r"matrix\(([-0-9\.eE]+),([-0-9\.eE]+),([-0-9\.eE]+),([-0-9\.eE]+),([-0-9\.eE]+),([-0-9\.eE]+)\)"
)

def _parse_transform(transform_str: str) -> np.ndarray:
    """
    Parse an SVG transform matrix string into a 3x3 numpy matrix.
    """
    m = _transform_re.search(transform_str)
    if not m:
        return np.eye(3)
    a, b, c, d, e, f = map(float, m.groups())
    return np.array([
        [a, c, e],
        [b, d, f],
        [0, 0, 1],
    ])


def _extract_coords(path_str: str) -> list[tuple[float, float]]:
    """
    Extract raw coordinates from a path string (ignores command letters).
    """
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", path_str)
    coords = []
    for i in range(0, len(numbers), 2):
        try:
            x = float(numbers[i])
            y = float(numbers[i + 1])
            coords.append((x, y))
        except IndexError:
            break
    return coords


def find_top_and_bottom(path_elem) -> tuple[float, float]:
    """
    Compute the top (min y) and bottom (max y) of an SVG path,
    taking transforms into account.
    Returns (top, bottom).
    """
    d = path_elem.get("d", "")
    coords = _extract_coords(d)
    if not coords:
        return 0, 0

    # Apply transform if available
    transform_attr = path_elem.get("transform", "")
    M = _parse_transform(transform_attr)

    transformed = []
    for (x, y) in coords:
        vec = np.array([x, y, 1.0])
        tx, ty, _ = M @ vec
        transformed.append((tx, ty))

    ys = [y for _, y in transformed]
    return min(ys), max(ys)

def looks_like_background(tag):
    if not hasattr(tag, 'name'):
        return False
    fill = (tag.get('fill') or '').lower()
    style = (tag.get('style') or '').lower()
    is_white_fill = any(x in fill for x in ('#fff', '#ffffff', 'white')) or 'fill:#fff' in style or 'fill:#ffffff' in style
    if not is_white_fill:
        return False
    if tag.name == 'path':
        d = tag.get('d', '')
        return d.strip().startswith('M0,0') or d.strip().startswith('M0 0')
    if tag.name == 'rect':
        x = tag.get('x') or '0'
        y = tag.get('y') or '0'
        try:
            return float(x) == 0.0 and float(y) == 0.0
        except Exception:
            return False
    return False

def merge_svgs_to_long_page(svg_paths):
    """
    Merge multiple SVG files into one very long vertical SVG and return the merged SVG as a string.
    """
    print("Merging SVGs to long page... ", end="", flush=True)
    svgs = []
    for p in svg_paths:
        with open(p, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")
        root = soup.find('svg')
        if root is None:
            raise ValueError(f"File {p} doesn't contain an <svg> root")

        vb = root.get('viewBox')
        if vb:
            parts = [float(x) for x in vb.split()]
            w, h = parts[2], parts[3]
        else:
            def num_attr(a):
                v = root.get(a)
                if not v:
                    return None
                m = re.match(r"([0-9.]+)", v)
                return float(m.group(1)) if m else None

            w = num_attr('width') or 0.0
            h = num_attr('height') or 0.0

        svgs.append({'path': p, 'width': w, 'height': h, 'soup': soup, 'root': root})

    if not svgs:
        raise ValueError('No svgs to merge')

    total_height = sum(s['height'] for s in svgs)
    max_width = max(s['width'] for s in svgs)
    combined_defs_parts = []
    page_elements = []
    single_title = None
    single_desc = None
    offset = 0.0

    for s in svgs:
        root = s['root']
        h = s['height']

        dtag = root.find('defs')
        if dtag:
            inner_defs = re.sub(r'^<defs>|</defs>$', '', str(dtag)).strip()
            combined_defs_parts.append(inner_defs)

        if single_title is None:
            t = root.find('title')
            if t:
                single_title = str(t)
        if single_desc is None:
            d = root.find('desc')
            if d:
                single_desc = str(d)
        top_level = [child for child in root.find_all(recursive=False)]

        elements_info = []
        for child in top_level:
            if child.name in ('defs', 'title', 'desc'):
                continue
            if looks_like_background(child):
                continue
            child_str = str(child)
            child_soup = BeautifulSoup(child_str, 'xml')
            child_tag = child_soup.find()
            y_top, y_bottom = find_top_and_bottom(child_tag)
            y_top += offset
            y_bottom += offset
            center = (y_top + y_bottom) / 2.0
            prev_t = child_tag.get('transform')
            translate = f'translate(0,{offset})'
            if prev_t:
                child_tag['transform'] = f"{translate} {prev_t}"
            else:
                child_tag['transform'] = translate
            elements_info.append({'orig': child, 'str': str(child_tag), 'y_top': y_top, 'y_bottom': y_bottom, 'center': center, 'soup': child_tag})
        for e in elements_info:
            page_elements.append(e['str'])
        offset += h
    merged_parts = ['<?xml version="1.0" encoding="utf-8"?>\n',
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(max_width)}" height="{int(total_height)}" viewBox="0 0 {max_width} {total_height}">']
    if single_title:
        merged_parts.append(single_title)
    if single_desc:
        merged_parts.append(single_desc)
    if combined_defs_parts:
        merged_parts.append('<defs>')
        merged_parts.extend(combined_defs_parts)
        merged_parts.append('</defs>')
    bg_d = f'M0,0 L{int(max_width)},0 L{int(max_width)},{int(total_height)} L0,{int(total_height)} L0,0 '
    merged_parts.append(f'<path xmlns="http://www.w3.org/2000/svg" d="{bg_d}" fill="#ffffff" fill-rule="evenodd"/>')
    merged_parts.extend(page_elements)
    merged_parts.append('</svg>')
    print("done")
    return '\n'.join(merged_parts)

def group_merged_svg_by_brackets(merged_svg_str: str):
    soup = BeautifulSoup(merged_svg_str, 'xml')
    root = soup.find('svg')
    if root is None:
        return merged_svg_str

    top_level = [child for child in root.find_all(recursive=False)]
    elems = []
    for child in top_level:
        if child.name in ('defs', 'title', 'desc'):
            continue
        if looks_like_background(child):
            continue
        y_top, y_bottom = find_top_and_bottom(child)
        center = (y_top + y_bottom) / 2.0
        elems.append({'orig': child, 'str': str(child), 'y_top': y_top, 'y_bottom': y_bottom, 'center': center})

    bracket_infos = []
    for ei in elems:
        cls = (ei['orig'].get('class') or '')
        if 'Bracket' in cls:
            top = ei['y_top']
            bottom = ei['y_bottom']
            bh = bottom - top
            bracket_infos.append({'info': ei, 'top': top, 'bottom': bottom, 'height': bh, 'center': (top + bottom) / 2.0})

    bracket_infos.sort(key=lambda x: x['center'])
    used = set()
    groups_info = []
    for idx, b in enumerate(bracket_infos):
        group_parts = []
        bracket_elem = b['info']
        group_parts.append((b['center'], bracket_elem['str']))
        used.add(id(bracket_elem['orig']))
        groups_info.append({'parts': group_parts, 'top': b['top'], 'bottom': b['bottom'], 'idx': idx})

    remaining = [e for e in elems if id(e['orig']) not in used]
    if groups_info:
        B = groups_info
        n = len(B)
        for e in remaining:
            y_top_e = e['y_top']
            y_bottom_e = e['y_bottom']
            y_center = e['center']
            assigned = False
            for g in B:
                if y_top_e <= g['bottom'] and y_bottom_e >= g['top']:
                    g['parts'].append((y_center, e['str']))
                    used.add(id(e['orig']))
                    assigned = True
                    break
            if assigned:
                continue
            if y_bottom_e < B[0]['top']:
                B[0]['parts'].append((y_center, e['str']))
                used.add(id(e['orig']))
                continue
            if y_top_e > B[-1]['bottom']:
                B[-1]['parts'].append((y_center, e['str']))
                used.add(id(e['orig']))
                continue
            for i in range(n - 1):
                above_bottom = B[i]['bottom']
                below_top = B[i+1]['top']
                if y_top_e >= above_bottom and y_bottom_e >= below_top:
                    d_above = y_top_e - above_bottom
                    d_below = below_top - y_bottom_e
                    if d_above <= d_below:
                        B[i]['parts'].append((y_center, e['str']))
                    else:
                        B[i+1]['parts'].append((y_center, e['str']))
                    used.add(id(e['orig']))
                    break

    grouped_parts = []
    if groups_info:
        for g in sorted(groups_info, key=lambda x: x['idx']):
            parts_sorted = [s for _, s in sorted(g['parts'], key=lambda t: t[0])]
            group_str = f'<g class="Group-{g["idx"]}">\n' + '\n'.join(parts_sorted) + '\n</g>'
            grouped_parts.append(group_str)
    else:
        if elems:
            grouped_parts.append('<g class="Fallback">\n' + '\n'.join(e['str'] for e in elems) + '\n</g>')

    header_parts = []
    if root.find('title'):
        header_parts.append(str(root.find('title')))
    if root.find('desc'):
        header_parts.append(str(root.find('desc')))
    dtag = root.find('defs')
    if dtag:
        header_parts.append(str(dtag))

    bg = None
    for child in root.find_all(recursive=False):
        child_tag = child.find()
        if looks_like_background(child_tag):
            bg = str(child)
            break

    xmlns = 'http://www.w3.org/2000/svg'
    max_width = root.get('width') or root.get('viewBox').split()[2] if root.get('viewBox') else root.get('width')
    total_height = root.get('height') or root.get('viewBox').split()[3] if root.get('viewBox') else root.get('height')

    merged_parts = ['<?xml version="1.0" encoding="utf-8"?>\n',
                    f'<svg xmlns="{xmlns}" width="{int(float(max_width)) if max_width else 0}" height="{int(float(total_height)) if total_height else 0}" viewBox="0 0 {max_width or 0} {total_height or 0}">']
    merged_parts.extend(header_parts)
    if bg:
        merged_parts.append(bg)
    merged_parts.extend(grouped_parts)
    merged_parts.append('</svg>')

    return '\n'.join(merged_parts)


def clean_and_write_merged_svg(merged_svg_str: str, out_path: Path, remove_classes=('Page', 'Tempo')):
    """
    Remove elements with any of the provided classes from the merged SVG string
    and write the cleaned SVG to out_path.
    """
    soup = BeautifulSoup(merged_svg_str, 'xml')
    selector = ', '.join('.' + c for c in remove_classes)
    for el in soup.select(selector):
        el.decompose()

    out_path = Path(out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    print('Cleaned merged SVG written to', out_path)


def midi_to_fixed_svg(midi_file: str, out_dir: str, mscore_cmd="mscore"):
    midi_path = Path(midi_file).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_svg = out_dir / (midi_path.stem + ".svg")
    run_musescore(midi_path, raw_svg, mscore_cmd=mscore_cmd)
    all_svgs = sorted(out_dir.glob(f"{midi_path.stem}*.svg"))
    pattern = re.compile(rf"^{re.escape(midi_path.stem)}(?:-\d+)?$")
    generated = [p for p in all_svgs if pattern.match(p.stem)]

    if not generated:
        raise FileNotFoundError(f"No SVG files found for {midi_path.stem} in {out_dir}")

    fixed_paths = []
    for g in generated:
        fixed = g.with_name(g.stem + g.suffix)
        print(f"Fixing: {g} -> {fixed}")
        fix_svg_scaling(g, fixed)
        fixed_paths.append(fixed)

    merged_svg_str = merge_svgs_to_long_page(fixed_paths)
    clean_and_write_merged_svg(merged_svg_str, out_dir/(midi_path.stem+"-pre.svg"), remove_classes=('Page', 'Tempo'))
    merged_svg_str = group_merged_svg_by_brackets(merged_svg_str)
    merged_out = out_dir / (midi_path.stem + ".svg")
    try:
        clean_and_write_merged_svg(merged_svg_str, merged_out, remove_classes=('Page', 'Tempo'))
    except Exception as e:
        print('Warning: failed to clean/write merged svg:', e)
    try:
        merged_resolved = merged_out.resolve()
        for p in fixed_paths:
            try:
                if p.resolve() != merged_resolved and p.exists():
                    p.unlink()
                    print('Deleted intermediate file', p)
            except Exception:
                pass
    except Exception:
        pass

    if len(fixed_paths) == 1:
        print("Done. Fixed SVG saved to:", fixed_paths[0])
    else:
        print("Done. Fixed SVGs processed:")
        for p in fixed_paths:
            print(" -", p)
        print('Final merged (cleaned) file:', merged_out)


if __name__ == "__main__":
    midi_file = "/home/u200b/Music/Credits Song For My Death.mid"
    out_dir = "./test/"
    mscore_cmd = "mscore"

    midi_to_fixed_svg(midi_file, out_dir, mscore_cmd)
