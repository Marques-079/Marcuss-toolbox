"""
Code from : https://github.com/Marques-079/Ai-plays-SubwaySurfers/alpha

This is what I used for Subway Surfers AI, setup on Macbook M1 pro, code accounts for the Retina display

./scgrab --x 644 --y 77 --w 505 --h 906 --fps 60 --out /tmp/scap.ring --slots 3 --scale 2


Uses ScreenCaptureKit to capture a rectangular region of your primary display.
Region is given in screen points: --x --y --w --h.
It scales the output by --scale and forces even width/height (required by NV12).
Captures at --fps, pixel format NV12 (video-range), no cursor, no audio.
Writes each frame into a memory-mapped ring buffer file at --out (default /tmp/scap.ring).
The ring has a GlobalHeader, then N slots (set by --slots). Each slot has a header + the packed NV12 frame (all Y rows, then all UV rows).
Prints a 1-second heartbeat: "[producer] wrote X frames in the last second".

--x --y --w --h : capture rectangle in points (not scaled)
--fps : frames per second (default 60)
--out : ring file path (default /tmp/scap.ring)
--slots : number of ring slots (default 3)
--scale : output pixel scale factor (default 2.0)
"""

from ring_grab import get_frame_bgr_from_ring
frame_bgr, meta = get_frame_bgr_from_ring(path="/tmp/scap.ring", wait_new=True)