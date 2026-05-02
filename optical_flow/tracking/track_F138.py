

import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import cv2
import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ── Defaults for this specific clip ────────────────────────────────────────────
DEFAULT_START = "0:13"
DEFAULT_END   = "0:17"
DEFAULT_OUT   = "./tracked_output"


# ═══════════════════════════════════════════════════════════════════════════════
# Time parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_time(t: str) -> float:
    """'MM:SS' -> seconds. '0:13' = 13s. Plain int/float also accepted."""
    s = str(t).strip()
    if not s:
        raise ValueError("empty time string")
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"expected MM:SS, got '{t}'")
        mm, ss = parts
        mm_i, ss_i = int(mm), int(ss)
        if not (0 <= ss_i < 60):
            raise ValueError(f"seconds must be 0..59 in '{t}'")
        return mm_i * 60 + ss_i
    return float(s)


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(round(sec - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}_{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing (annotated video only)
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 221,  51), (100, 221,  23),
    ( 24, 201, 114), ( 24, 201, 201), ( 23, 114, 255), (125,  12, 255),
    (255,  12, 240), (255, 120, 120), (198, 255, 120), (120, 255, 198),
    (120, 198, 255), (198, 120, 255), (255, 198, 120), (180, 180, 180),
]

def color_for(tid: int):
    return PALETTE[int(tid) % len(PALETTE)]

def draw_box(frame, x1, y1, x2, y2, tid, conf, interpolated=False):
    color = color_for(tid)
    thickness = 1 if interpolated else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    prefix = "~" if interpolated else ""
    label = f"{prefix}ID:{tid} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Tracker config
# ═══════════════════════════════════════════════════════════════════════════════

def write_botsort_cfg(path: Path):
    """BoT-SORT tuned for short fight clips: minimize ID fragmentation."""
    cfg = """\
tracker_type: botsort

track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.60

track_buffer: 150

match_thresh: 0.90
fuse_score: True

gmc_method: sparseOptFlow

proximity_thresh: 0.30
appearance_thresh: 0.25
with_reid: True
model: auto
"""
    path.write_text(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Gap interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_gaps(raw_tracks: dict, max_gap: int = 45) -> dict:
    """Linearly interpolate missing frames across gaps <= max_gap frames."""
    filled = {}
    for tid, frame_dict in raw_tracks.items():
        filled[tid] = dict(frame_dict)
        frames = sorted(frame_dict.keys())
        for i in range(len(frames) - 1):
            f_a, f_b = frames[i], frames[i + 1]
            gap = f_b - f_a - 1
            if gap < 1 or gap > max_gap:
                continue
            b_a = np.array(frame_dict[f_a][:4], dtype=float)
            b_b = np.array(frame_dict[f_b][:4], dtype=float)
            avg_conf = (frame_dict[f_a][4] + frame_dict[f_b][4]) / 2.0
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                box = ((1 - t) * b_a + t * b_b).astype(int)
                filled[tid][f_a + step] = (
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                    float(avg_conf),
                )
    return filled


# ═══════════════════════════════════════════════════════════════════════════════
# CVAT-for-video 1.1 XML export  (BUG-FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

def write_cvat_xml(xml_path: Path,
                   filled_tracks: dict,
                   raw_tracks: dict,
                   n_frames: int,
                   width: int,
                   height: int,
                   video_name: str):
    """
    CVAT-for-video 1.1 annotations.

    Bug fixes relative to the original track_clip.py:

      * <name> instead of <n>: the legacy <n> tag is rejected by CVAT's
        project-save path.
      * Terminator box rule: only written when the last visible frame is
        strictly before n_frames-1. Writing outside="1" on top of the last
        visible box was producing two boxes on the same frame in one track,
        which caused CVAT save 500.

    Attributes:
      - role:           mutable=True, radio(non_aggressor|aggressor),
                        default non_aggressor. Flip per-frame inside CVAT.
      - uncertain_clip: mutable=False, checkbox, default false.
    """
    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append("<annotations>")
    lines.append("  <version>1.1</version>")
    lines.append("  <meta>")
    lines.append("    <task>")
    lines.append(f"      <size>{n_frames}</size>")
    lines.append("      <mode>interpolation</mode>")
    lines.append("      <overlap>0</overlap>")
    lines.append(f'      <original_size><width>{width}</width><height>{height}</height></original_size>')
    lines.append("      <labels>")
    lines.append("        <label>")
    lines.append("          <name>person</name>")
    lines.append("          <attributes>")
    lines.append("            <attribute>")
    lines.append("              <name>role</name>")
    lines.append("              <mutable>True</mutable>")
    lines.append("              <input_type>radio</input_type>")
    lines.append("              <default_value>non_aggressor</default_value>")
    lines.append("              <values>non_aggressor\naggressor</values>")
    lines.append("            </attribute>")
    lines.append("            <attribute>")
    lines.append("              <name>uncertain_clip</name>")
    lines.append("              <mutable>False</mutable>")
    lines.append("              <input_type>checkbox</input_type>")
    lines.append("              <default_value>false</default_value>")
    lines.append("              <values>false</values>")
    lines.append("            </attribute>")
    lines.append("          </attributes>")
    lines.append("        </label>")
    lines.append("      </labels>")
    lines.append("    </task>")
    lines.append(f"    <source>{xml_escape(video_name)}</source>")
    lines.append("  </meta>")

    def clamp_xyxy(x1, y1, x2, y2):
        x1 = max(0, min(int(x1), width  - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width  - 1))
        y2 = max(0, min(int(y2), height - 1))
        return x1, y1, x2, y2

    for track_idx, (tid, fdict) in enumerate(sorted(filled_tracks.items())):
        if not fdict:
            continue
        lines.append(f'  <track id="{track_idx}" label="person" source="manual">')
        original_frames = set(raw_tracks.get(tid, {}).keys())
        sorted_frames   = sorted(fdict.keys())

        for f in sorted_frames:
            x1, y1, x2, y2, conf = fdict[f]
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            is_interp = f not in original_frames
            occluded  = 1 if is_interp else 0
            lines.append(
                f'    <box frame="{f}" outside="0" occluded="{occluded}" '
                f'keyframe="1" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0">'
            )
            lines.append('      <attribute name="role">non_aggressor</attribute>')
            lines.append('      <attribute name="uncertain_clip">false</attribute>')
            lines.append('    </box>')

        # ── Terminator only when there is room past the last visible frame ──
        last_f = sorted_frames[-1]
        if last_f < n_frames - 1:
            term_f = last_f + 1
            x1, y1, x2, y2, _ = fdict[last_f]
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2)
            lines.append(
                f'    <box frame="{term_f}" outside="1" occluded="0" '
                f'keyframe="1" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0">'
            )
            lines.append('      <attribute name="role">non_aggressor</attribute>')
            lines.append('      <attribute name="uncertain_clip">false</attribute>')
            lines.append('    </box>')
        # else: track runs to the final frame of the clip; no terminator.

        lines.append("  </track>")

    lines.append("</annotations>")
    xml_path.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════════════════════

def process_clip(video_path: Path, start_sec: float, end_sec: float,
                 tracked_path: Path, clean_path: Path, xml_path: Path,
                 model: YOLO, tracker_cfg: Path,
                 conf_thr: float = 0.15, max_gap: int = 45) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total / fps if fps > 0 else 0

    if start_sec < 0:
        start_sec = 0.0
    if end_sec > duration_sec:
        print(f"  [warn] end={end_sec:.2f}s clipped to duration {duration_sec:.2f}s")
        end_sec = duration_sec
    if end_sec <= start_sec:
        raise ValueError(f"end ({end_sec}) must be > start ({start_sec})")

    start_frame = int(round(start_sec * fps))
    end_frame   = int(round(end_sec   * fps))   # inclusive
    n_frames    = end_frame - start_frame + 1

    print(f"  video: {width}x{height} @ {fps:.2f} fps, {total} frames "
          f"({duration_sec:.2f}s total)")
    print(f"  slice: frames [{start_frame}..{end_frame}] = {n_frames} frames "
          f"({start_sec:.2f}s -> {end_sec:.2f}s)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_tracked = cv2.VideoWriter(str(tracked_path), fourcc, fps, (width, height))
    writer_clean   = cv2.VideoWriter(str(clean_path),   fourcc, fps, (width, height))

    # ── Pass 1: detect + track ────────────────────────────────────────────────
    raw_frames = []
    raw_tracks = defaultdict(dict)

    print(f"  Pass 1/2 - detecting + tracking ...", flush=True)
    local_idx = 0
    while local_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"  [warn] video ended early at local frame {local_idx}")
            break
        raw_frames.append(frame)
        writer_clean.write(frame)

        track_res = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=conf_thr,
            iou=0.40,
            tracker=str(tracker_cfg),
            verbose=False,
        )
        boxes = track_res[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy  = boxes.xyxy.cpu().numpy().astype(int)
            tids  = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, conf in zip(xyxy, tids, confs):
                raw_tracks[int(tid)][local_idx] = (
                    int(x1), int(y1), int(x2), int(y2), float(conf)
                )

        local_idx += 1
        if local_idx % 30 == 0:
            print(f"    {local_idx}/{n_frames} frames processed", flush=True)

    cap.release()
    writer_clean.release()
    actual_n_frames = len(raw_frames)

    # ── Gap interpolation ─────────────────────────────────────────────────────
    print(f"  Interpolating short gaps (<= {max_gap} frames) ...", flush=True)
    filled = interpolate_gaps(dict(raw_tracks), max_gap=max_gap)

    per_frame = defaultdict(list)
    for tid, fdict in filled.items():
        original = set(raw_tracks[tid].keys())
        for f, (x1, y1, x2, y2, conf) in fdict.items():
            per_frame[f].append((tid, x1, y1, x2, y2, conf, f not in original))

    # ── Pass 2: render annotated video ────────────────────────────────────────
    print(f"  Pass 2/2 - rendering annotated video ...", flush=True)
    for f_local, frame in enumerate(raw_frames):
        out = frame.copy()
        for tid, x1, y1, x2, y2, conf, interp in per_frame.get(f_local, []):
            draw_box(out, x1, y1, x2, y2, tid, conf, interpolated=interp)
        h, w = out.shape[:2]
        t_sec = start_sec + f_local / fps
        hud = f"t={t_sec:.2f}s  frame {start_frame + f_local}  (local {f_local})"
        cv2.putText(out, hud, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer_tracked.write(out)
    writer_tracked.release()

    # ── CVAT XML ──────────────────────────────────────────────────────────────
    print(f"  Writing CVAT XML ...", flush=True)
    write_cvat_xml(
        xml_path=xml_path,
        filled_tracks=filled,
        raw_tracks=raw_tracks,
        n_frames=actual_n_frames,
        width=width,
        height=height,
        video_name=clean_path.name,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {}
    for tid, fdict in filled.items():
        orig  = sorted(raw_tracks[tid].keys())
        all_f = sorted(fdict.keys())
        summary[tid] = {
            "first_local_frame":   all_f[0]  if all_f else -1,
            "last_local_frame":    all_f[-1] if all_f else -1,
            "frames_detected":     len(orig),
            "frames_interpolated": len(all_f) - len(orig),
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bounding-box tracker + CVAT XML exporter for F_138_0_0_0_0.mp4"
    )
    parser.add_argument("--video", required=True, help="Path to F_138_0_0_0_0.mp4")
    parser.add_argument("--start", default=DEFAULT_START, help="Start MM:SS (default 0:13)")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End MM:SS (default 0:17)")
    parser.add_argument("--output_dir", default=DEFAULT_OUT,
                        help="Output directory (default ./tracked_output)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO weights (default yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Detection confidence threshold (default 0.15)")
    parser.add_argument("--max_gap", type=int, default=45,
                        help="Max frame gap to interpolate (default 45, ~1.5s @ 30fps)")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"[ERROR] Video not found: {video_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    tracker_cfg = output_dir / "botsort_tuned.yaml"
    write_botsort_cfg(tracker_cfg)

    print(f"Loading YOLO: {args.model}")
    model = YOLO(args.model)

    start_sec = parse_time(args.start)
    end_sec   = parse_time(args.end)

    out_stem = f"{video_path.stem}_{fmt_time(start_sec)}_to_{fmt_time(end_sec)}"
    tracked_path = output_dir / f"{out_stem}_tracked.mp4"
    clean_path   = output_dir / f"{out_stem}_clean.mp4"
    xml_path     = output_dir / f"{out_stem}_cvat.xml"

    print(f"Processing: {video_path.name}  [{start_sec:.2f}s -> {end_sec:.2f}s]")
    summary = process_clip(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        tracked_path=tracked_path,
        clean_path=clean_path,
        xml_path=xml_path,
        model=model,
        tracker_cfg=tracker_cfg,
        conf_thr=args.conf,
        max_gap=args.max_gap,
    )

    print("Wrote:")
    print(f"  tracked: {tracked_path}")
    print(f"  clean:   {clean_path}")
    print(f"  cvat:    {xml_path}")
    print(f"Tracks found: {len(summary)}")
    for tid, info in sorted(summary.items()):
        print(f"  ID {tid}: frames {info['first_local_frame']}..{info['last_local_frame']} "
              f"(detected={info['frames_detected']}, "
              f"interpolated={info['frames_interpolated']})")


if __name__ == "__main__":
    main()
