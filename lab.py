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

    if len(fixed_paths) == 1:
        print("Done. Fixed SVG saved to:", fixed_paths[0])
    else:
        print("Done. Fixed SVGs saved to:")
        for p in fixed_paths:
            print(" -", p)


if __name__ == "__main__":
    midi_file = "/home/u200b/Music/Credits Song For My Death.mid"
    out_dir = "./test/"
    mscore_cmd = "mscore"

    midi_to_fixed_svg(midi_file, out_dir, mscore_cmd)
