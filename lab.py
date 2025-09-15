#!/usr/bin/env python3
import subprocess
import re
from pathlib import Path
from bs4 import BeautifulSoup

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


def merge_svgs_to_long_page(svg_paths):
    """
    Merge multiple SVG files into one very long vertical SVG and return
    the merged SVG as a string (do not write to disk here).

    This version does not wrap each page in its own <g>. Instead it
    translates each top-level element by the appropriate Y offset so
    all elements live in a single coordinate space. It also preserves
    only one <title> and <desc> (from the first SVG) and combines <defs>.
    Any per-page background paths/rects (detected by fill white and
    coordinates starting at the top-left) are skipped and replaced by
    a single big background covering the full merged area.
    """
    svgs = []
    for p in svg_paths:
        with open(p, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")
        root = soup.find('svg')
        if root is None:
            raise ValueError(f"File {p} doesn't contain an <svg> root")

        # Determine width/height using viewBox if available
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

    # Collect combined defs, single title/desc and page elements
    combined_defs_parts = []
    page_elements = []
    single_title = None
    single_desc = None
    offset = 0.0

    def looks_like_background(tag, w, h):
        # Heuristic: path starting at M0,0 or rect at 0,0 with white fill
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

    for s in svgs:
        root = s['root']
        w = s['width']
        h = s['height']

        # collect defs
        dtag = root.find('defs')
        if dtag:
            # strip outer <defs> and store inner text
            inner_defs = re.sub(r'^<defs>|</defs>$', '', str(dtag)).strip()
            combined_defs_parts.append(inner_defs)

        # title/desc only from first svg
        if single_title is None:
            t = root.find('title')
            if t:
                single_title = str(t)
        if single_desc is None:
            d = root.find('desc')
            if d:
                single_desc = str(d)

        # iterate top-level children and translate them into merged coordinate space
        for child in root.find_all(recursive=False):
            # skip defs/title/desc handled above
            if child.name in ('defs', 'title', 'desc'):
                continue

            # skip backgrounds from individual pages
            if looks_like_background(child, w, h):
                continue

            # make a copy by converting to string and reparsing to avoid cross-soup issues
            child_str = str(child)
            child_soup = BeautifulSoup(child_str, 'xml')
            child_tag = child_soup.find()

            # adjust/prepend transform to translate by current offset
            prev_t = child_tag.get('transform')
            translate = f'translate(0,{offset})'
            if prev_t:
                # prepend translate so existing transform still applies
                child_tag['transform'] = f"{translate} {prev_t}"
            else:
                child_tag['transform'] = translate

            page_elements.append(str(child_tag))

        offset += h

    # Build merged SVG string
    header = '<?xml version="1.0" encoding="utf-8"?>\n'
    xmlns = 'http://www.w3.org/2000/svg'
    merged_parts = [header]
    merged_parts.append(f'<svg xmlns="{xmlns}" width="{int(max_width)}" height="{int(total_height)}" viewBox="0 0 {max_width} {total_height}">')

    # single title/desc
    if single_title:
        merged_parts.append(single_title)
    if single_desc:
        merged_parts.append(single_desc)

    # combined defs
    if combined_defs_parts:
        merged_parts.append('<defs>')
        merged_parts.extend(combined_defs_parts)
        merged_parts.append('</defs>')

    # single big background (path) covering the whole merged area
    bg_d = f'M0,0 L{int(max_width)},0 L{int(max_width)},{int(total_height)} L0,{int(total_height)} L0,0 '
    merged_parts.append(f'<path xmlns="http://www.w3.org/2000/svg" d="{bg_d}" fill="#ffffff" fill-rule="evenodd"/>')

    # add all page elements (they already have transforms applied)
    merged_parts.extend(page_elements)

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

    # Merge the fixed svgs into one long SVG page (get string, don't write here)
    merged_svg_str = merge_svgs_to_long_page(fixed_paths)
    merged_out = out_dir / (midi_path.stem + ".svg")

    # Clean removed classes and write the final merged svg
    try:
        clean_and_write_merged_svg(merged_svg_str, merged_out, remove_classes=('Page', 'Tempo'))
    except Exception as e:
        print('Warning: failed to clean/write merged svg:', e)

    # Delete the individual fixed files (keep the merged_out)
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
