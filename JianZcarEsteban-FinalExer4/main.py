import cv2
import numpy as np

input_path = "input.mp4"
output_path = "4_output.mp4"

cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

scan_line_spacing = 4
scan_speed = 1  # pixels per frame

for i in range(N):
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    green_channel = cv2.addWeighted(gray_frame, 1.2, np.zeros_like(gray_frame), 0, 10)

    noise = np.random.normal(0, 10, (h, w)).astype(np.int16)
    noisy_green = np.clip(green_channel.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    output_frame = np.zeros((h, w, 3), dtype=np.uint8)
    output_frame[:, :, 1] = noisy_green

    scan_offset = (i * scan_speed) % scan_line_spacing
    for y in range(scan_offset, h, scan_line_spacing):
        output_frame[y:y+1, :] = (output_frame[y:y+1, :] * 0.4).astype(np.uint8)

    out.write(output_frame)

cap.release()
out.release()
print("Done")
