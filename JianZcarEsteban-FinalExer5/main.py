import cv2
import numpy as np

input_path = "input.mp4"
output_path = "5_output.mp4"

cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

x = np.linspace(-1, 1, w)
y = np.linspace(-1, 1, h)
x_grid, y_grid = np.meshgrid(x, y)
d = np.sqrt(x_grid**2 + y_grid**2)

base_sigma = 0.5
static_mask = np.exp(-d**2 / (2 * base_sigma**2))
static_mask = np.clip(static_mask, 0, 1)

mask_3ch = np.stack([static_mask]*3, axis=-1)

for i in range(N):
    ret, frame = cap.read()
    if not ret:
        break

    pulse_strength = 0.9 + 0.1 * np.sin(2 * np.pi * i / 90)  # adjust 90 for pulse speed
    pulsating_mask = np.clip(mask_3ch * pulse_strength, 0, 1)

    frame_float = frame.astype(np.float32)
    vignette_frame = (frame_float * pulsating_mask).clip(0, 255).astype(np.uint8)

    out.write(vignette_frame)

cap.release()
out.release()
print("Done applying vignette effect.")
