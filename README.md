# Upper Body Extraction Tool using Mediapipe

A tool I developed to extract upper body frames from videos using MediaPipe face detection and pose estimation.

## Research Purpose

I'm conducting research on generating upper body movement using diffusion models. This tool helps me create a high-quality dataset for training various state-of-the-art animation models such as:

- [MooreAnimate](https://github.com/snap-research/articulated-animation)
- [MagicDance](https://boese0601.github.io/magicdance/)
- [LIA (Live Image Animation)](https://wyhsirius.github.io/LIA-project/)
- [MimicMotion](https://tencent.github.io/MimicMotion/)
- [Articulated Animation](https://snap-research.github.io/articulated-animation/)

The goal is to create a robust dataset of human upper body movements that can be used to train models that generate realistic human animations from still images.

## How It Works

I use a combination of face detection and pose estimation to:
1. Find faces in the video frame
2. Detect upper body pose landmarks
3. Keep only the frames where both face and upper body are properly visible

Through experimentation, I found that faces should occupy between 1.80% and 4.20% of the frame area for optimal upper body visibility. I also check that important landmarks (head, shoulders, arms) are clearly visible with a visibility score above 0.5.

The script works with videos of any resolution, as it uses relative measurements (percentages) rather than absolute pixel values. This makes it versatile for processing videos from different sources and qualities.

## Example Results

I've tested the tool on various videos, including talks from 99U:

Original videos:
- [Video 1: Original Source](https://www.youtube.com/watch?v=ayQeYjSVd3Y&ab_channel=99U)
- [Video 2: Original Source](https://www.youtube.com/watch?v=R75AfAjsTiQ&ab_channel=99U)

### Video Demo

Since GitHub doesn't directly support playing videos in README files, you can check out these animated GIFs showing the processing in action:

![Video 1 GIF](/upperbody_exraction_mediapipe/GIFs/video1.gif)
![Video 2 GIF](/upperbody_exraction_mediapipe/GIFs/video2gif)

**Note:** These example GIFs and videos are just short clips demonstrating the processing technique, not the complete processed output of the entire YouTube videos. They're meant to illustrate how the tool works rather than provide the full dataset.

## Installation

```
pip install -r requirements.txt
```

Make sure you have ffmpeg installed:
```
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

Process videos by pointing to your input and output folders:

```
python video_processor.py --input /path/to/videos --output /path/to/results
```

### Options

- `--input` or `-i`: Folder containing video files
- `--output` or `-o`: Where to save processed videos
- `--processes` or `-p`: Number of parallel processes (default: auto)

## Creating Your Own Dataset

If you want to create a similar dataset:

1. Collect YouTube video IDs of interest
2. Download videos using youtube-dl or similar tools:
   ```
   youtube-dl -f 'bestvideo[height<=720]' -o '%(id)s.%(ext)s' VIDEO_ID
   ```
3. Run this tool to extract good quality upper body frames
4. Use the processed videos for your machine learning projects

## Requirements

- Python 3.6+
- ffmpeg-python
- numpy
- mediapipe
- opencv-python
- psutil