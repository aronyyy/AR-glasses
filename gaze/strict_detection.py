import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt

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
            current_ratio = max(length / width, width / length)
            if current_ratio <= ratio_thresh and area > max_area:
                max_area = area
                largest_contour = contour

    return [largest_contour] if largest_contour is not None else []


def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0, 0, 0]
    if len(contour) < 5:
        return 0

    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)
    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    if ellipse_area == 0:
        return ellipse_goodness
    ellipse_goodness[0] = covered_pixels / ellipse_area
    ellipse_goodness[2] = min(ellipse[1][1] / ellipse[1][0], ellipse[1][0] / ellipse[1][1])
    return ellipse_goodness


def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0]

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
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]


def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours
    all_contours = np.concatenate(contours[0], axis=0)
    spacing = int(len(all_contours) / 25)
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        with np.errstate(invalid='ignore'):
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60))
        if np.dot(vec_to_centroid, (vec1 + vec2) / 2) >= cos_threshold:
            filtered_points.append(current_point)
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


def process_frames(thresholded_image_strict, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    final_rotated_rect = ((0, 0), (0, 0), 0)
    kernel = np.ones((5, 5), np.uint8)
    gray_copy = gray_frame.copy()

    dilated_image = cv2.dilate(thresholded_image_strict, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
        current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
        total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)
        ellipse = cv2.fitEllipse(reduced_contours[0])
        cv2.ellipse(gray_copy, ellipse, (255, 0, 0), 2)
        if debug_mode_on:
            cv2.imshow("Strict Threshold", gray_copy)

        final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]
        test_frame = frame.copy()
        final_contours = [optimize_contours_by_angle(reduced_contours, gray_frame)]
        if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
            ellipse = cv2.fitEllipse(final_contours[0])
            final_rotated_rect = ellipse
            cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
            center_x, center_y = map(int, ellipse[0])
            cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
            cv2.putText(test_frame, "SPACE = play/pause", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 90, 30), 2)
            cv2.putText(test_frame, "Q      = quit", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 90, 30), 2)
        if render_cv_window:
            cv2.imshow('Strict threshold result', test_frame)

    return final_rotated_rect


def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 1)
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)
    final_rotated_rect = process_frames(thresholded_image_strict, frame, gray_frame, darkest_point, False, False)
    return final_rotated_rect


def process_video(video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    debug_mode_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect_ratio(frame)
        darkest_point = get_darkest_area(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        thresholded_image_strict = mask_outside_square(apply_binary_threshold(gray_frame, darkest_pixel_value, 1), darkest_point, 250)

        pupil_rotated_rect = process_frames(thresholded_image_strict, frame, gray_frame, darkest_point, debug_mode_on, True)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        print("No file selected. Exiting.")
        return
    process_video(video_path)


if __name__ == "__main__":
    select_video()
