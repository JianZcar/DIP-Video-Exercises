import cv2
import numpy as np
import math

# 1. Open input video and prepare output
input_path = 'input.mp4'
output_path = 'output.mp4'

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

frame_index = 0

# Ask the user for the effect type
effect_type = input("Choose the effect type ('linear' or 'pulse'): "
                    ).strip().lower()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    b, g, r = cv2.split(frame)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    gray = gray.astype(np.uint8)
    progress = frame_index / total_frames

    if effect_type == 'linear':
        alpha = 0.8 + (1.5 - 0.8) * progress
    elif effect_type == 'pulse':
        alpha = 0.8 + 0.7 * math.sin(2 * math.pi * 4 * progress)
    else:
        print("Invalid choice. Defaulting to linear effect.")
        alpha = 0.8 + (1.5 - 0.8) * progress

    mean = np.mean(gray)
    adjusted = alpha * (gray - mean) + mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    output_frame = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)

    out.write(output_frame)

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
