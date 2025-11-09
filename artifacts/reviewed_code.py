# Video Editing Workflow Script for Plan Summary
# Output: artifacts/output_video.mp4

import os
from pathlib import Path

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
    vfx,
    ColorClip,
)
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from pydub import AudioSegment

# Ensure output and temp directories exist
ARTIFACTS_DIR = Path("artifacts")
TEMP_DIR = ARTIFACTS_DIR / "temp"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def apply_brightness_contrast(clip, brightness=1.0, contrast=1.0):
    def bc(image):
        img = image.astype(np.float32)
        img = img * brightness
        mean = img.mean(axis=(0,1), keepdims=True)
        img = (img - mean) * contrast + mean
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    return clip.fl_image(bc)

def apply_saturation(clip, saturation=1.0):
    def sat(image):
        img = image.astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[...,1] = np.clip(hsv[...,1] * saturation, 0, 255)
        img2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img2
    return clip.fl_image(sat)

def apply_blur(clip, ksize=3):
    def blur(image):
        return cv2.GaussianBlur(image, (ksize|1, ksize|1), 0)
    return clip.fl_image(blur)

def apply_sepia(clip, intensity=30):
    def sepia(image):
        img = image.astype(np.float32)
        tr = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
        sep = img @ tr.T
        img = img + (sep - img) * (intensity/100.0)
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    return clip.fl_image(sepia)

def apply_vignette(clip, strength=1):
    def vignette(image):
        rows, cols = image.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        mask = mask ** (1.5 * strength)
        out = image.astype(np.float32)
        for i in range(3):
            out[...,i] *= mask
        out = np.clip(out, 0, 255)
        return out.astype(np.uint8)
    return clip.fl_image(vignette)

def apply_stabilization(clip):
    print("  [Stabilize] Running basic stabilization (may be slow)...")
    frames = []
    for frame in tqdm(list(clip.iter_frames()), desc="  [Stabilize]"):
        frames.append(frame)
    stabilized_frames = stabilize_frames(frames)
    def make_frame(t):
        idx = int(t * clip.fps)
        if idx >= len(stabilized_frames):
            idx = len(stabilized_frames)-1
        return stabilized_frames[idx]
    return clip.set_make_frame(make_frame)

def stabilize_frames(frames):
    transforms = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        assert prev_pts is not None and curr_pts is not None
        idx = np.where(status==1)[0]
        prev_pts_good = prev_pts[idx]
        curr_pts_good = curr_pts[idx]
        m = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)[0]
        if m is None:
            m = np.eye(2,3)
        transforms.append(m)
        prev_gray = curr_gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    trajectory = np.cumsum([m[:,2] for m in transforms], axis=0)
    smoothed = smooth_trajectory(trajectory)
    diff = smoothed - trajectory
    stabilized = []
    for i, frame in enumerate(frames):
        if i==0:
            stabilized.append(frame)
            continue
        m = transforms[i-1].copy()
        m[:,2] += diff[i-1]
        stabilized_frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        stabilized.append(stabilized_frame)
    return stabilized

def smooth_trajectory(trajectory, radius=10):
    smoothed = np.copy(trajectory)
    for i in range(len(trajectory)):
        start = max(0, i-radius)
        end = min(len(trajectory), i+radius+1)
        smoothed[i] = np.mean(trajectory[start:end], axis=0)
    return smoothed

def add_text_overlay(clip, text, position, fontsize, color, bg_color=None, bg_opacity=0.5, start=0, end=None):
    txt_clip = TextClip(
        text,
        fontsize=fontsize,
        color=color,
        font="Arial-Bold",
        method='label'
    )
    if bg_color:
        w, h = txt_clip.size
        bg = ColorClip(size=(w, h), color=(0,0,0)).set_opacity(bg_opacity)
        txt_clip = CompositeVideoClip([bg, txt_clip.set_position('center')], size=(w, h))
    txt_clip = txt_clip.set_position(position).set_start(start).set_duration((end or clip.duration) - start)
    return CompositeVideoClip([clip, txt_clip.set_duration(clip.duration)])

def add_text_overlay_custom(clip, text, position, fontsize, color, bg_color=None, bg_opacity=0.5, start=0, end=None):
    txt_clip = TextClip(
        text,
        fontsize=fontsize,
        color=color,
        font="Arial-Bold",
        method='label'
    )
    if bg_color:
        w, h = txt_clip.size
        bg = ColorClip(size=(w, h), color=(0,0,0)).set_opacity(bg_opacity)
        txt_clip = CompositeVideoClip([bg, txt_clip.set_position('center')], size=(w, h))
    if position == 'center':
        pos = ('center', 'center')
    elif position == 'bottom center':
        pos = ('center', 'bottom')
    elif position == 'top left':
        pos = ('left', 'top')
    else:
        pos = position
    txt_clip = txt_clip.set_position(pos).set_start(start).set_duration((end or clip.duration) - start)
    return CompositeVideoClip([clip, txt_clip.set_duration(clip.duration)])

def process_audio_keep(clip, start, end, volume=1.0, fadein=0.0, fadeout=0.0):
    duration = end - start
    if clip.audio is None:
        # Add a larger buffer to avoid OSError at the end
        buffer_ms = 1000  # 1 second buffer
        silent = AudioSegment.silent(duration=int(duration*1000 + buffer_ms))
        temp_silence = TEMP_DIR / f"keep_silence_{duration:.2f}.wav"
        silent.export(str(temp_silence), format="wav")
        # Ensure subclip does not exceed the actual duration of the silent audio
        audio_clip = AudioFileClip(str(temp_silence))
        max_duration = audio_clip.duration
        subclip_end = min(duration, max_duration - 0.01)  # leave a small margin
        audio = audio_clip.subclip(0, subclip_end)
        if fadein > 0:
            audio = audio.audio_fadein(fadein)
        if fadeout > 0:
            audio = audio.audio_fadeout(fadeout)
        audio = audio.volumex(volume)
        return audio
    else:
        audio = clip.audio.subclip(start, end)
        if fadein > 0:
            audio = audio.audio_fadein(fadein)
        if fadeout > 0:
            audio = audio.audio_fadeout(fadeout)
        audio = audio.volumex(volume)
        return audio

def process_audio_replace(clip, audio_path, duration, handling='trim', volume=1.0):
    if audio_path == 'DEFAULT: silence':
        # Add a larger buffer to avoid OSError at the end
        buffer_ms = 1000  # 1 second buffer
        silent = AudioSegment.silent(duration=int(duration*1000 + buffer_ms))
        temp_silence = TEMP_DIR / f"silence_{duration:.2f}.wav"
        silent.export(str(temp_silence), format="wav")
        audio_clip = AudioFileClip(str(temp_silence))
        max_duration = audio_clip.duration
        subclip_end = min(duration, max_duration - 0.01)  # leave a small margin
        audio = audio_clip.subclip(0, subclip_end)
        audio = audio.volumex(volume)
        return audio
    else:
        audio = AudioFileClip(audio_path)
        if handling == 'loop':
            n_loops = int(np.ceil(duration / audio.duration))
            audios = [audio] * n_loops
            audio = concatenate_audioclips(audios).subclip(0, duration)
        else:
            audio = audio.subclip(0, duration)
        audio = audio.volumex(volume)
        return audio

def concatenate_audioclips(clips):
    from moviepy.audio.AudioClip import concatenate_audioclips
    return concatenate_audioclips(clips)

def apply_wipe_transition(clip1, clip2, duration=1.0):
    w, h = clip1.size
    def make_frame(t):
        if t < duration:
            alpha = t / duration
            x = int(w * alpha)
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            frame = frame1.copy()
            frame[:, :x] = frame2[:, :x]
            return frame
        else:
            return clip2.get_frame(t - duration)
    new_duration = duration + clip2.duration
    new_clip = clip1.set_duration(duration).set_end(duration)
    wipe_clip = VideoFileClip(clip2.filename).set_start(duration).set_duration(clip2.duration)
    composite = VideoFileClip(clip1.filename).set_duration(duration)
    from moviepy.video.VideoClip import VideoClip
    wipe = VideoClip(make_frame=make_frame, duration=new_duration)
    wipe = wipe.set_fps(clip1.fps)
    wipe = wipe.set_audio(clip2.audio)
    return wipe

def process_clip_1():
    print("Processing Clip 1...")
    src = "workspace/video36.mp4"
    clip = VideoFileClip(src).subclip(0.0, 10.0)
    clip = clip.fx(vfx.fadein, 0.5)
    clip = apply_brightness_contrast(clip, brightness=1.1, contrast=1.2)
    clip = add_text_overlay_custom(
        clip,
        text="Welcome to Our Video",
        position='center',
        fontsize=48,
        color='white',
        bg_color='black',
        bg_opacity=0.5,
        start=0.0,
        end=3.0
    )
    audio = process_audio_keep(clip, 0.0, 10.0, volume=0.8, fadein=0.5, fadeout=0.5)
    clip = clip.set_audio(audio)
    return clip

def process_clip_2():
    print("Processing Clip 2...")
    src = "workspace/video37.mp4"
    clip = VideoFileClip(src).subclip(0.0, 8.5)
    clip = clip.fx(vfx.speedx, 1.2)
    clip = apply_stabilization(clip)
    audio = process_audio_replace(
        clip,
        audio_path="workspace/ff-16b-2c-44100hz.mp3",
        duration=clip.duration,
        handling='loop',
        volume=1.0
    )
    clip = clip.set_audio(audio)
    return clip

def process_clip_3():
    print("Processing Clip 3...")
    src = "workspace/video38.mp4"
    clip = VideoFileClip(src).subclip(2.0, 12.0)
    clip = apply_sepia(clip, intensity=30)
    clip = apply_vignette(clip, strength=1)
    clip = add_text_overlay_custom(
        clip,
        text="Scene Transition",
        position='bottom center',
        fontsize=36,
        color='yellow',
        bg_color=None,
        start=1.0,
        end=4.0
    )
    audio = process_audio_keep(clip, 0.0, 10.0, volume=0.7, fadein=0.3, fadeout=0.0)
    clip = clip.set_audio(audio)
    return clip

def process_clip_4():
    print("Processing Clip 4...")
    src = "workspace/video52.mp4"
    clip = VideoFileClip(src).subclip(0.0, 6.0)
    clip = apply_blur(clip, ksize=3)
    clip = clip.fx(vfx.mirror_x)
    audio = process_audio_replace(
        clip,
        audio_path="workspace/file_example_MP3_1MG.mp3",
        duration=clip.duration,
        handling='trim',
        volume=0.9
    )
    clip = clip.set_audio(audio)
    return clip

def process_clip_5():
    print("Processing Clip 5...")
    src = "workspace/video53.mp4"
    clip = VideoFileClip(src).subclip(0.0, 9.5)
    clip = apply_brightness_contrast(clip, brightness=0.9, contrast=1.1)
    clip = apply_saturation(clip, saturation=1.2)
    clip = add_text_overlay_custom(
        clip,
        text="Thank You",
        position='top left',
        fontsize=24,
        color='red',
        bg_color=None,
        start=0.0,
        end=2.0
    )
    audio = process_audio_replace(
        clip,
        audio_path="DEFAULT: silence",
        duration=clip.duration,
        handling='trim',
        volume=0.0
    )
    clip = clip.set_audio(audio)
    def rotate90(image):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    clip = clip.fl_image(rotate90)
    w, h = clip.size
    clip = clip.resize(0.8)
    clip = clip.fx(vfx.fadeout, 1.0)
    return clip

def main():
    print("Starting video editing workflow...")

    clip1 = process_clip_1()
    clip2 = process_clip_2()
    clip3 = process_clip_3()
    clip4 = process_clip_4()
    clip5 = process_clip_5()

    print("Applying between-clip transitions...")

    print("  [Transition] 1-2: crossfade (1.0s)")
    crossfade_duration = 1.0
    clip1 = clip1.set_end(clip1.duration)
    clip2 = clip2.set_start(clip1.duration - crossfade_duration)
    clip1 = clip1.crossfadeout(crossfade_duration)
    clip2 = clip2.crossfadein(crossfade_duration)

    print("  [Transition] 2-3: wipe (0.8s)")
    wipe_duration = 0.8
    clip2_main = clip2.subclip(0, clip2.duration - wipe_duration)
    clip3_main = clip3.subclip(wipe_duration, clip3.duration)
    clip2_end = clip2.subclip(clip2.duration - wipe_duration, clip2.duration)
    clip3_start = clip3.subclip(0, wipe_duration)
    if clip2_end.size != clip3_start.size:
        min_w = min(clip2_end.size[0], clip3_start.size[0])
        min_h = min(clip2_end.size[1], clip3_start.size[1])
        clip2_end = clip2_end.resize((min_w, min_h))
        clip3_start = clip3_start.resize((min_w, min_h))
    def wipe_make_frame(t):
        alpha = t / wipe_duration
        x = int(clip2_end.size[0] * alpha)
        frame2 = clip2_end.get_frame(t)
        frame3 = clip3_start.get_frame(t)
        frame = frame2.copy()
        frame[:, :x] = frame3[:, :x]
        return frame
    from moviepy.video.VideoClip import VideoClip
    wipe_clip = VideoClip(make_frame=wipe_make_frame, duration=wipe_duration)
    wipe_clip = wipe_clip.set_fps(clip2.fps)
    wipe_audio = clip2_end.audio.audio_fadeout(wipe_duration).audio_fadein(wipe_duration)
    wipe_clip = wipe_clip.set_audio(wipe_audio)

    print("Concatenating all clips...")
    final_clips = [
        clip1,
        clip2_main,
        wipe_clip,
        clip3_main,
        clip4,
        clip5
    ]
    final = concatenate_videoclips(final_clips, method="compose")

    output_path = ARTIFACTS_DIR / "output_video.mp4"
    print(f"Exporting final video to {output_path} ...")
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(TEMP_DIR / "temp-audio.m4a"),
        remove_temp=True,
        threads=4,
        fps=24
    )
    print("Done.")

if __name__ == "__main__":
    main()