# Complete executable Python script implementing the requested video editing plan
# Requirements:
# - MoviePy 1.0.3
# - NumPy, OpenCV, Pillow (optional), imageio, scipy (optional), pydub (optional), tqdm (optional)
#
# Notes:
# - This script avoids try/except to let errors surface
# - Uses OpenCV-based text rendering to avoid external ImageMagick dependency for TextClip
# - Implements custom wipe transition (left_to_right) with proper audio crossfade
# - Applies specified effects, audio handling, and transitions
#
# Usage:
#   Ensure the source files exist under "workspace/" as specified, then run this script.

from pathlib import Path
import numpy as np
import cv2

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    vfx,
    afx,
)

# ---------- Utility: filesystem ----------
def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# ---------- Utility: colors and positions ----------
def color_name_to_rgb(name):
    name = str(name).lower()
    mapping = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    if name in mapping:
        return mapping[name]
    # default to white if unknown
    return (255, 255, 255)

def compute_position_xy(position, overlay_w, overlay_h, base_w, base_h):
    margin = int(round(0.04 * min(base_w, base_h)))  # 4% margin
    pos = str(position).lower()
    if pos in ("center", "centre"):
        x = (base_w - overlay_w) // 2
        y = (base_h - overlay_h) // 2
    elif pos in ("bottom_center", "bottom_centre"):
        x = (base_w - overlay_w) // 2
        y = base_h - overlay_h - margin
    elif pos in ("top_left", "topleft"):
        x = margin
        y = margin
    else:
        # default to center
        x = (base_w - overlay_w) // 2
        y = (base_h - overlay_h) // 2
    return x, y

# ---------- Text overlay via OpenCV (no ImageMagick dependency) ----------
def build_text_rgba_image(
    text,
    fontsize=48,
    text_color="white",
    bg_color=None,
    bg_opacity=0.5,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    # Determine fontScale to approximate desired pixel height = fontsize
    # getTextSize returns ((w,h), baseline) for scale=1.0 use as baseline
    base_thickness = max(1, int(round(fontsize / 12)))
    ((w1, h1), base1) = cv2.getTextSize(text, font, 1.0, base_thickness)
    if h1 == 0:
        h1 = 1
    fontScale = float(fontsize) / float(h1)
    thickness = base_thickness

    ((text_w, text_h), baseline) = cv2.getTextSize(text, font, fontScale, thickness)
    pad = int(round(fontsize * 0.5))
    overall_w = max(2, text_w + 2 * pad)
    overall_h = max(2, text_h + 2 * pad)

    rgba = np.zeros((overall_h, overall_w, 4), dtype=np.uint8)

    # Background rectangle alpha and color
    if bg_color is not None:
        bg_rgb = color_name_to_rgb(bg_color)
        rgba[..., 0] = bg_rgb[0]
        rgba[..., 1] = bg_rgb[1]
        rgba[..., 2] = bg_rgb[2]
        rgba[..., 3] = int(round(np.clip(bg_opacity, 0.0, 1.0) * 255))
    else:
        # Transparent background
        rgba[..., 3] = 0

    # Create text alpha mask
    text_mask = np.zeros((overall_h, overall_w), dtype=np.uint8)
    text_origin = (pad, pad + text_h)  # baseline -> y increases downward
    cv2.putText(
        text_mask,
        text,
        text_origin,
        font,
        fontScale,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Apply text color on RGB channels where mask > 0
    tc = color_name_to_rgb(text_color)
    for c in range(3):
        channel = rgba[..., c]
        channel[text_mask > 0] = tc[c]
        rgba[..., c] = channel

    # Make text fully opaque
    alpha = rgba[..., 3]
    alpha[text_mask > 0] = 255
    rgba[..., 3] = alpha

    return rgba

def make_text_overlay_clip(
    base_size,  # (w,h)
    text,
    position="center",
    fontsize=48,
    color="white",
    bg_color=None,
    bg_opacity=0.5,
    start=0.0,
    end=3.0,
):
    w, h = base_size
    rgba = build_text_rgba_image(
        text=text,
        fontsize=fontsize,
        text_color=color,
        bg_color=bg_color,
        bg_opacity=bg_opacity,
    )
    rgb = rgba[..., :3]
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    mask_clip = ImageClip(alpha, ismask=True)
    img_clip = ImageClip(rgb).set_mask(mask_clip)

    # Compute position
    ov_h, ov_w = rgb.shape[0], rgb.shape[1]
    x, y = compute_position_xy(position, ov_w, ov_h, w, h)
    img_clip = img_clip.set_position((x, y)).set_start(start).set_duration(max(0.0, end - start))
    return img_clip

# ---------- Video effect utilities ----------
def make_brightness_contrast_filter(brightness=1.0, contrast=1.0):
    # brightness multiplicative, contrast around mid-gray 128
    b = float(brightness)
    c = float(contrast)
    def _f(frame):
        # frame is RGB uint8
        f = frame.astype(np.float32)
        # apply brightness multiplicatively
        f = f * b
        # apply contrast around 128
        f = (f - 128.0) * c + 128.0
        f = np.clip(f, 0, 255).astype(np.uint8)
        return f
    return _f

def make_saturation_filter(saturation=1.0):
    s = float(saturation)
    def _f(frame):
        f = frame.astype(np.uint8)
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        hsv = hsv.astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return out
    return _f

def make_blur_filter(sigma=3):
    sig = float(sigma)
    def _f(frame):
        # Use Gaussian blur, sigma in pixels
        out = cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_DEFAULT)
        return out
    return _f

def make_sepia_vignette_filter(sepia_intensity=0.3, vignette_strength=0.5, size=None):
    # Precompute mask based on size (w,h)
    # If size None, will compute dynamically on first call
    sepia_t = float(np.clip(sepia_intensity, 0.0, 1.0))
    vig_t = float(np.clip(vignette_strength, 0.0, 1.0))
    cache = {"mask": None, "shape": None}

    # Sepia matrix for RGB
    sepia_kernel = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    def _make_mask(h, w):
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        X, Y = np.meshgrid(xx, yy)
        r = np.sqrt(X * X + Y * Y)  # 0 at center to ~1.414 at corners
        # Normalize to 0..1
        r = r / np.sqrt(2.0)
        # Vignette mask: multiplier from 1 at center downwards towards edges
        mask = 1.0 - vig_t * (r ** 2)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        return mask

    def _f(frame):
        h, w = frame.shape[0], frame.shape[1]
        if (cache["mask"] is None) or (cache["shape"] != (h, w)):
            cache["mask"] = _make_mask(h, w)
            cache["shape"] = (h, w)
        mask = cache["mask"]

        f = frame.astype(np.float32)
        # Sepia transform on RGB
        sepia = f @ sepia_kernel.T
        sepia = np.clip(sepia, 0, 255)

        # Blend sepia with original by sepia_intensity
        blended = (1.0 - sepia_t) * f + sepia_t * sepia

        # Apply vignette (multiply each channel by mask)
        blended *= mask[..., None]
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended
    return _f

# ---------- Transitions: wipe mask ----------
def make_wipe_mask_clip(duration, size, direction="left_to_right"):
    w, h = size
    d = float(duration)
    dirn = str(direction).lower()

    def make_frame(t):
        p = float(np.clip(t / d, 0.0, 1.0))
        if dirn in ("left_to_right", "ltr", "lefttoright"):
            x_thresh = int(round(p * w))
            mask = np.zeros((h, w), dtype=np.float32)
            if x_thresh > 0:
                mask[:, :x_thresh] = 1.0
            return mask
        elif dirn in ("right_to_left", "rtl", "righttoleft"):
            x_thresh = int(round((1.0 - p) * w))
            mask = np.zeros((h, w), dtype=np.float32)
            if x_thresh < w:
                mask[:, x_thresh:] = 1.0
            return mask
        elif dirn in ("top_to_bottom", "ttb"):
            y_thresh = int(round(p * h))
            mask = np.zeros((h, w), dtype=np.float32)
            if y_thresh > 0:
                mask[:y_thresh, :] = 1.0
            return mask
        elif dirn in ("bottom_to_top", "btt"):
            y_thresh = int(round((1.0 - p) * h))
            mask = np.zeros((h, w), dtype=np.float32)
            if y_thresh < h:
                mask[y_thresh:, :] = 1.0
            return mask
        else:
            # default left_to_right
            x_thresh = int(round(p * w))
            mask = np.zeros((h, w), dtype=np.float32)
            if x_thresh > 0:
                mask[:, :x_thresh] = 1.0
            return mask

    mask_clip = ImageClip(make_frame, ismask=True, duration=d)
    return mask_clip

# ---------- Audio helpers ----------
def apply_keep_audio(clip, window_start, window_end, volume, fadein=0.0, fadeout=0.0):
    if clip.audio is None:
        return clip
    win_start = float(window_start)
    win_end = float(window_end)
    sub = clip.audio.subclip(win_start, min(win_end, clip.duration))
    if volume is not None:
        sub = sub.volumex(float(volume))
    if fadein and fadein > 0:
        sub = sub.fx(afx.audio_fadein, float(fadein))
    if fadeout and fadeout > 0:
        sub = sub.fx(afx.audio_fadeout, float(fadeout))
    return clip.set_audio(sub)

def apply_replace_audio(clip, audio_path, window_start, window_end, handling="loop", external_volume=1.0):
    ext = AudioFileClip(str(audio_path))
    win_start = float(window_start)
    win_end = float(window_end)
    base = ext.subclip(win_start, win_end)
    if str(handling).lower() == "loop":
        new_audio = afx.audio_loop(base, duration=clip.duration)
    else:
        # trim: just cut to the clip duration window
        new_audio = base.subclip(0, min(base.duration, clip.duration))
    if external_volume is not None:
        new_audio = new_audio.volumex(float(external_volume))
    return clip.set_audio(new_audio)

# ---------- Timeline assembly with transitions ----------
def assemble_timeline(clips, base_size, transitions):
    # clips: list of processed VideoFileClip-like objects (effects applied, no set_start yet)
    # transitions: list of dicts for between-clip transitions; len = len(clips)-1
    # Returns CompositeVideoClip
    print("Assembling timeline with transitions...")
    timeline_elements = []

    # Start with the first clip at t=0
    t = 0.0
    c_prev = clips[0]
    timeline_elements.append(c_prev.set_start(t))
    t_end_prev = t + c_prev.duration

    for i, trans in enumerate(transitions):
        c_next = clips[i + 1]
        ttype = str(trans.get("type", "cut")).lower()
        if ttype == "crossfade":
            d = float(trans.get("duration", 1.0))
            print(f" - Transition {i+1}->{i+2}: crossfade {d:.2f}s")

            # Apply audio crossfade between c_prev and c_next
            c_prev = c_prev.fx(afx.audio_fadeout, d)
            c_next = c_next.fx(afx.audio_fadein, d)

            start_next = t_end_prev - d
            timeline_elements.append(c_next.set_start(start_next))
            # Update end time
            t_end_prev = start_next + c_next.duration
            # Replace prev reference for subsequent transitions
            c_prev = c_next

        elif ttype == "wipe":
            d = float(trans.get("duration", 0.8))
            direction = trans.get("direction", "left_to_right")
            print(f" - Transition {i+1}->{i+2}: wipe {d:.2f}s direction={direction}")

            # Audio crossfade
            c_prev = c_prev.fx(afx.audio_fadeout, d)
            c_next = c_next.fx(afx.audio_fadein, d)

            # Overlap region
            overlap_start = t_end_prev - d
            mask = make_wipe_mask_clip(d, base_size, direction=direction)

            c_next_overlap = c_next.subclip(0, min(d, c_next.duration)).set_mask(mask).set_start(overlap_start)
            timeline_elements.append(c_next_overlap)

            # Remaining part after wipe
            if c_next.duration > d:
                c_next_after = c_next.subclip(d, c_next.duration).set_start(t_end_prev)
                timeline_elements.append(c_next_after)

            # Update end time
            t_end_prev = overlap_start + d + max(0.0, c_next.duration - d)
            c_prev = c_next

        elif ttype in ("cut", "none", "no", "straight"):
            print(f" - Transition {i+1}->{i+2}: cut")
            start_next = t_end_prev
            timeline_elements.append(c_next.set_start(start_next))
            t_end_prev = start_next + c_next.duration
            c_prev = c_next
        else:
            print(f" - Transition {i+1}->{i+2}: unknown '{ttype}', defaulting to cut")
            start_next = t_end_prev
            timeline_elements.append(c_next.set_start(start_next))
            t_end_prev = start_next + c_next.duration
            c_prev = c_next

    final = CompositeVideoClip(timeline_elements, size=(base_size[0], base_size[1]))
    return final

# ---------- Main processing per clip ----------
def main():
    print("Starting video editing workflow...")

    artifacts_dir = Path("artifacts")
    ensure_dir(artifacts_dir)
    output_path = artifacts_dir / "output_video.mp4"

    # Sources
    src1 = Path("workspace/video36.mp4")
    src2 = Path("workspace/video37.mp4")
    src3 = Path("workspace/video38.mp4")
    src4 = Path("workspace/video52.mp4")
    src5 = Path("workspace/video53.mp4")

    # Load and process Clip 1
    print("Processing Clip 1...")
    clip1 = VideoFileClip(str(src1)).subclip(0.0, 10.0)
    # Effects: brightness, contrast
    clip1 = clip1.fl_image(make_brightness_contrast_filter(brightness=1.1, contrast=1.2))
    # Text overlay
    print(" - Adding text overlay to Clip 1")
    base_size1 = (clip1.w, clip1.h)
    overlay1 = make_text_overlay_clip(
        base_size1,
        text="Welcome to Our Video",
        position="center",
        fontsize=48,
        color="white",
        bg_color="black",
        bg_opacity=0.5,
        start=0.0,
        end=3.0,
    )
    clip1 = CompositeVideoClip([clip1, overlay1], size=base_size1)
    # Audio: keep, volume, fades
    clip1 = apply_keep_audio(clip1, window_start=0.0, window_end=10.0, volume=0.8, fadein=0.5, fadeout=0.5)
    # Pre-transition fade in (in-place edge)
    clip1 = clip1.fx(vfx.fadein, 0.5)

    # Load and process Clip 2
    print("Processing Clip 2...")
    clip2 = VideoFileClip(str(src2)).subclip(0.0, 8.5)
    # Effects: speed=1.2x, stabilize default skip
    print(" - Applying speedx 1.2x to Clip 2")
    clip2 = clip2.fx(vfx.speedx, 1.2)
    print(" - Stabilization skipped (default=1.0)")
    # Audio: replace with external looped/trimmed
    print(" - Replacing audio for Clip 2")
    clip2 = clip2.set_audio(None)
    clip2 = apply_replace_audio(
        clip2,
        audio_path=Path("workspace/ff-16b-2c-44100hz.mp3"),
        window_start=0.0,
        window_end=7.0833,
        handling="loop",
        external_volume=1.0,
    )

    # Load and process Clip 3
    print("Processing Clip 3...")
    clip3 = VideoFileClip(str(src3)).subclip(2.0, 12.0)
    # Effects: sepia + vignette
    print(" - Applying sepia and vignette to Clip 3")
    clip3 = clip3.fl_image(make_sepia_vignette_filter(sepia_intensity=0.3, vignette_strength=0.5))
    # Text overlay
    print(" - Adding text overlay to Clip 3")
    base_size3 = (clip3.w, clip3.h)
    overlay3 = make_text_overlay_clip(
        base_size3,
        text="Scene Transition",
        position="bottom_center",
        fontsize=36,
        color="yellow",
        bg_color=None,
        bg_opacity=0.0,
        start=1.0,
        end=4.0,
    )
    clip3 = CompositeVideoClip([clip3, overlay3], size=base_size3)
    # Audio: keep, volume 0.7, fade in 0.3
    clip3 = apply_keep_audio(clip3, window_start=0.0, window_end=10.0, volume=0.7, fadein=0.3, fadeout=0.0)
    # Additional ops: rotate=0, scale=1.0 (no change)

    # Load and process Clip 4
    print("Processing Clip 4...")
    clip4 = VideoFileClip(str(src4)).subclip(0.0, 6.0)
    # Effects: blur=3
    print(" - Applying blur sigma=3 to Clip 4")
    clip4 = clip4.fl_image(make_blur_filter(sigma=3))
    # Audio: replace with external trimmed
    print(" - Replacing audio for Clip 4 (trim)")
    clip4 = clip4.set_audio(None)
    clip4 = apply_replace_audio(
        clip4,
        audio_path=Path("workspace/file_example_MP3_1MG.mp3"),
        window_start=0.0,
        window_end=6.0,
        handling="trim",
        external_volume=0.9,
    )
    # Additional ops: flip horizontal
    clip4 = clip4.fx(vfx.mirror_x)

    # Load and process Clip 5
    print("Processing Clip 5...")
    clip5 = VideoFileClip(str(src5)).subclip(0.0, 9.5)
    # Effects: brightness=0.9, contrast=1.1, saturation=1.2
    print(" - Applying brightness/contrast/saturation to Clip 5")
    clip5 = clip5.fl_image(make_brightness_contrast_filter(brightness=0.9, contrast=1.1))
    clip5 = clip5.fl_image(make_saturation_filter(saturation=1.2))
    # Text overlay
    print(" - Adding text overlay to Clip 5")
    base_size5 = (clip5.w, clip5.h)
    overlay5 = make_text_overlay_clip(
        base_size5,
        text="Thank You",
        position="top_left",
        fontsize=24,
        color="red",
        bg_color=None,
        bg_opacity=0.0,
        start=0.0,
        end=2.0,
    )
    clip5 = CompositeVideoClip([clip5, overlay5], size=base_size5)
    # Audio: keep, but original_volume=0.0 -> mute
    clip5 = clip5.set_audio(clip5.audio.volumex(0.0) if clip5.audio is not None else None)
    # Additional ops: rotate=90, scale=0.8
    clip5 = clip5.rotate(90)
    clip5 = clip5.resize(0.8)
    # Post-transition fade out (in-place edge)
    clip5 = clip5.fx(vfx.fadeout, 1.0)

    # Determine base output size from first clip after processing
    base_w, base_h = clip1.w, clip1.h
    base_size = (base_w, base_h)
    print(f"Base output size set to {base_w}x{base_h}")

    # Normalize sizes of all clips to base size (to ensure seamless transitions)
    def resize_to_base(c):
        if (c.w, c.h) != base_size:
            return c.resize(newsize=base_size)
        return c

    clip1 = resize_to_base(clip1)
    clip2 = resize_to_base(clip2)
    clip3 = resize_to_base(clip3)
    clip4 = resize_to_base(clip4)
    clip5 = resize_to_base(clip5)

    # Between-clip transitions specification
    transitions = [
        {"type": "crossfade", "duration": 1.0},                         # 1 -> 2
        {"type": "wipe", "duration": 0.8, "direction": "left_to_right"},# 2 -> 3
        {"type": "cut"},                                                # 3 -> 4
        {"type": "none"},                                               # 4 -> 5
    ]

    # Assemble final composite timeline
    final = assemble_timeline([clip1, clip2, clip3, clip4, clip5], base_size, transitions)

    # Determine FPS from first clip if available, else default 30
    fps_out = clip1.fps if hasattr(clip1, "fps") and clip1.fps is not None else 30
    print(f"Exporting final video to: {output_path} at {fps_out} fps")

    # Export final video
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps_out,
        preset="medium",
        threads=4,
        temp_audiofile=str(artifacts_dir / "temp-audio.m4a"),
        remove_temp=True,
        verbose=True,
    )

    # Close clips to release resources
    final.close()
    clip1.close()
    clip2.close()
    clip3.close()
    clip4.close()
    clip5.close()
    print("Done.")

if __name__ == "__main__":
    main()