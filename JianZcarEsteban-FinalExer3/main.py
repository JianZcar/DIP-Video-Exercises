import cv2
import math

input_path = "input.mp4"
output_path = "3_output.mp4"
final_angle = 360

cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

center = (w / 2, h / 2)


def compute_scale_to_fit(w, h, angle_degrees):
    angle = math.radians(angle_degrees)
    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))
    new_w = w * cos_a + h * sin_a
    new_h = w * sin_a + h * cos_a
    scale_w = w / new_w
    scale_h = h / new_h
    return min(scale_w, scale_h)


for i in range(N):
    ret, frame = cap.read()
    if not ret:
        break

    current_angle = (final_angle * i) / (N - 1) if N > 1 else final_angle
    scale = compute_scale_to_fit(w, h, current_angle)
    M = cv2.getRotationMatrix2D(center, current_angle, scale)
    rotated = cv2.warpAffine(frame, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    out.write(rotated)

cap.release()
out.release()
print("Done rotating video.")
