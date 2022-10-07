import os
from glob import glob
import moviepy.video.io.ImageSequenceClip
image_files = sorted(glob("../img_concat_full/*.png"))
fps=10
duration = 120

frame = fps * duration

if len(image_files) > frame:
    image_files = image_files[:frame]

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('/home/toytiny/SAIC_radar/radar_comp_concat.mp4')