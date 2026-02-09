import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
from tkinter import filedialog
import sys
import time

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

# ==========================
# Load fixed eyeball center
# ==========================

EYEBALL_CENTER_FILE = "eye_calib_eyeball_center.txt"  # change to your calib filename if needed

def load_eyeball_center(path=EYEBALL_CENTER_FILE, default_res=(640, 480)):
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
        x_str, y_str = line.split(",")
        return (int(x_str), int(y_str))
    except Exception as e:
        print("Could not load eyeball center from file, falling back to image center:", e)
        return (default_res[0] // 2, default_res[1] // 2)

FIXED_EYEBALL_CENTER = load_eyeball_center()

# ==========================
# Existing helpers (copied from vector.py, unchanged)
# ==========================

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)

    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)
            if width == 0:
                continue
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)
            if current_ratio <= ratio_thresh and area > max_area:
                max_area = area
                largest_contour = contour

    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0, None]

    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)

    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)

    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10)
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4)

    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)

    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)

    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = (
        absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    )

    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0, 0, 0]
    if len(contour) < 5:
        return ellipse_goodness

    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)

    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))

    if ellipse_area == 0:
        return ellipse_goodness

    ellipse_goodness[0] = covered_pixels / ellipse_area

    axes_lengths = ellipse[1]
    if axes_lengths[0] == 0 or axes_lengths[1] == 0:
        ellipse_goodness[2] = 0
    else:
        ellipse_goodness[2] = min(
            axes_lengths[1] / axes_lengths[0], axes_lengths[0] / axes_lengths[1]
        )

    return ellipse_goodness

# ==========================
# Gaze vector computation (same as your vector.py)
# ==========================

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    viewport_width = screen_width
    viewport_height = screen_height

    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    camera_position = np.array([0.0, 0.0, 3.0])

    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    origin = ray_origin
    direction = -ray_direction

    L = origin - sphere_center
    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t_candidates = [t for t in [t1, t2] if t > 0]
        if not t_candidates:
            return None, None
        t = min(t_candidates)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)

    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_ * x_ * x_ + c,       t_ * x_ * y_ - s * z_, t_ * x_ * z_ + s * y_],
        [t_ * x_ * y_ + s * z_,  t_ * y_ * y_ + c,      t_ * y_ * z_ - s * x_],
        [t_ * x_ * z_ - s * y_,  t_ * y_ * z_ + s * x_, t_ * z_ * z_ + c     ],
    ])

    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    file_path = "gaze_vector.txt"

    def is_file_available(path):
        try:
            with open(path, "a"):
                return True
        except IOError:
            return False

    if is_file_available(file_path):
        try:
            with open(file_path, "w") as f:
                all_values = np.concatenate((sphere_center, gaze_rotated))
                csv_line = ",".join(f"{v:.6f}" for v in all_values)
                f.write(csv_line + "\n")
        except Exception as e:
            print("Write error:", e)
    else:
        print("File is currently in use. Skipping write.")

    return sphere_center, gaze_rotated

# ==========================
# NEW process_frames (uses FIXED_EYEBALL_CENTER)
# ==========================

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed,
                   frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    """
    Detect pupil ellipse; do NOT estimate eyeball center.
    Use FIXED_EYEBALL_CENTER from calibration file to compute gaze vector.
    """
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict]
    name_array = ["relaxed", "medium", "strict"]
    gray_copies = [gray_frame.copy(), gray_frame.copy(), gray_frame.copy()]

    best_ellipse = None
    center_x, center_y = None, None
    goodness = 0

    for i in range(3):
        dilated_image = cv2.dilate(image_array[i], kernel, iterations=2)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)
            final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]

            if final_goodness > goodness:
                goodness = final_goodness
                best_ellipse = ellipse

    eyeball_center = FIXED_EYEBALL_CENTER

    if best_ellipse is not None:
        (center_x, center_y), axes, angle = best_ellipse
        center_x, center_y = int(center_x), int(center_y)

        # Draw fixed eyeball center and pupil ellipse
        cv2.circle(frame, eyeball_center, 8, (255, 255, 0), -1)  # eyeball center
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)  # pupil center
        cv2.ellipse(frame, best_ellipse, (20, 255, 255), 2)

        # line from eyeball center to pupil center
        cv2.line(frame, eyeball_center, (center_x, center_y), (255, 150, 50), 2)

        # extended gaze line
        dx = center_x - eyeball_center[0]
        dy = center_y - eyeball_center[1]
        extended_x = int(eyeball_center[0] + 2 * dx)
        extended_y = int(eyeball_center[1] + 2 * dy)
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)

        # compute gaze vector with fixed eyeball center
        sphere_center, direction = compute_gaze_vector(
            center_x, center_y,
            eyeball_center[0], eyeball_center[1],
            screen_width=frame.shape[1],
            screen_height=frame.shape[0]
        )

        if sphere_center is not None and direction is not None:
            origin_text = f"Origin: ({sphere_center[0]:.2f}, {sphere_center[1]:.2f}, {sphere_center[2]:.2f})"
            dir_text = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
            text_origin = (12, frame.shape[0] - 38)
            text_dir = (12, frame.shape[0] - 13)
            text_origin2 = (10, frame.shape[0] - 40)
            text_dir2 = (10, frame.shape[0] - 15)

            cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if GL_SPHERE_AVAILABLE:
                gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, eyeball_center[0], eyeball_center[1])
                if gl_image is not None:
                    blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
                    cv2.imshow("Eye Tracker + Sphere", blended)
                else:
                    cv2.imshow("Eye Tracker + Sphere", frame)
            else:
                cv2.imshow("Eye Tracker + Sphere", frame)

        else:
            cv2.imshow("Eye Tracker + Sphere", frame)

    else:
        # No ellipse found; just show frame
        cv2.imshow("Eye Tracker + Sphere", frame)

    return best_ellipse, sphere_center if 'sphere_center' in locals() else None, \
           direction if 'direction' in locals() else None

# ==========================
# Your process_frame + GUI hooks (using new process_frames)
# ==========================

# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):
    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)
    frame = cv2.flip(frame, -1)

    # find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

    # apply thresholding operations at different levels
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)  # lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)  # medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)

    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)  # heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

    # take the three images thresholded at different levels and process them
    result = process_frames(
        thresholded_image_strict,
        thresholded_image_medium,
        thresholded_image_relaxed,
        frame,
        gray_frame,
        darkest_point,
        False,
        False
    )

    # Normalize return to (ellipse, sphere_center, direction)
    if isinstance(result, tuple) and len(result) == 3:
        final_rotated_rect, sphere_center, direction = result
    else:
        final_rotated_rect, sphere_center, direction = None, None, None

    return final_rotated_rect, sphere_center, direction

# Process video from the selected camera
def process_camera():
    global selected_camera
    cam_index = int(selected_camera.get())

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 0)
        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# Process a selected video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# GUI for selecting video
def selection_gui():
    root = tk.Tk()
    root.withdraw()  # hide main window

    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv")]
    )

    if not video_path:
        print("No file selected. Exiting...")
        return

    process_video(video_path)

if __name__ == "__main__":
    selection_gui()
