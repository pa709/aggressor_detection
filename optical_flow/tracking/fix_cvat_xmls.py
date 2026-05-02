

import sys
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def fix_xml(xml_path: Path) -> tuple[int, int]:
    """Returns (n_terminators_removed, n_tag_renames)."""
    # Read raw to preserve formatting
    content = xml_path.read_text()

    # Fix bug 2 & 3: rename <n> to <name> and flip mutable True -> False on role
    # Only the label-definition <n> tags; attribute <attribute name="..."> tags inside
    # <box> already use name= attribute correctly
    renames = 0
    # Match <n>xxx</n> tags (these only exist in the bad schema)
    new_content, count = re.subn(r'<n>(.*?)</n>', r'<name>\1</name>', content)
    renames += count

    # Flip mutable True -> False for role (we made everything non-mutable)
    new_content = re.sub(
        r'(<name>role</name>\s*<mutable>)True(</mutable>)',
        r'\1False\2',
        new_content,
    )

    # Now parse and fix bug 1
    root = ET.fromstring(new_content)
    terminators_removed = 0

    for track in root.findall("track"):
        boxes = list(track.findall("box"))
        if not boxes:
            continue
        # Group boxes by frame
        frame_to_boxes = {}
        for b in boxes:
            f = int(b.attrib["frame"])
            frame_to_boxes.setdefault(f, []).append(b)

        # Find duplicates: a frame with both a visible (outside=0) and terminator (outside=1)
        for f, bs in frame_to_boxes.items():
            if len(bs) > 1:
                # Keep the visible one, drop the terminator
                visible = [b for b in bs if b.attrib.get("outside", "0") == "0"]
                terminator = [b for b in bs if b.attrib.get("outside", "0") == "1"]
                if visible and terminator:
                    for t in terminator:
                        track.remove(t)
                        terminators_removed += 1

    # Write back with XML declaration
    out = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )
    xml_path.write_text(out)

    return terminators_removed, renames


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 fix_cvat_xmls.py <dir>")
        sys.exit(1)

    d = Path(sys.argv[1])
    if not d.is_dir():
        print(f"Not a directory: {d}")
        sys.exit(1)

    xml_files = sorted(d.glob("*_cvat.xml"))
    print(f"Found {len(xml_files)} CVAT XML files in {d}")
    print()

    total_term = 0
    total_ren = 0
    for x in xml_files:
        term, ren = fix_xml(x)
        print(f"  {x.name}: removed {term} terminator dup(s), renamed {ren} tag(s)")
        total_term += term
        total_ren += ren

    print()
    print(f"Total: {total_term} terminator duplicates removed, "
          f"{total_ren} <n> tags renamed to <name>")


if __name__ == "__main__":
    main()
