import cv2
import numpy as np

input_path = "input.mp4"
output_path = "2_output.mp4"
kernel_size = 21
blur_region_width = 100
smooth_blending = True

cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    cx = (frame_width / total_frames) * i
    blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    x_start = max(0, int(cx - blur_region_width / 2))
    x_end = min(frame_width, int(cx + blur_region_width / 2))

    output_frame = frame.copy()

    if smooth_blending:
        alpha = np.zeros((frame_height, x_end - x_start), dtype=np.float32)
        half_width = (x_end - x_start) // 2

        for x in range(x_end - x_start):
            if x < half_width:
                alpha[:, x] = x / half_width
            else:
                alpha[:, x] = 1 - (x - half_width) / half_width
        alpha = np.clip(alpha, 0, 1)
        alpha = cv2.merge([alpha] * 3)  # Match 3 channels

        blurred_region = blurred_frame[:, x_start:x_end].astype(np.float32)
        original_region = frame[:, x_start:x_end].astype(np.float32)
        blended = (alpha * blurred_region + (1 - alpha) * original_region
                   ).astype(np.uint8)
        output_frame[:, x_start:x_end] = blended
    else:
        # Hard cut
        output_frame[:, x_start:x_end] = blurred_frame[:, x_start:x_end]

    out.write(output_frame)

cap.release()
out.release()
print("Done")
