# Cell 1: Imports
import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt

# ============================================================
# Cell 2: Image Preprocessing Functions

def crop_to_aspect_ratio(image, width=640, height=480):
    """Crop the image to maintain a specific aspect ratio (width:height) before resizing."""
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))


def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    """Apply thresholding to an image."""
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image


def mask_outside_square(image, center, size):
    """Mask all pixels outside a square defined by center and size."""
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# ============================================================
# Cell 3: Dark Region Detection

def get_darkest_area(image):
    """
    Finds a square area of dark pixels in the image.
    Returns a point within the pupil region.
    """
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area
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

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

# ============================================================
# Cell 4: Contour Optimization Functions

def optimize_contours_by_angle(contours, image):
    """Optimize contours by filtering points based on angle criteria."""
    if len(contours) < 1:
        return contours

    all_contours = np.concatenate(contours[0], axis=0)
    spacing = int(len(all_contours)/25)
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    point_image = image.copy()
    
    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        
        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60))
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))


def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    """
    Returns the largest contour that is not extremely long or tall.
    """
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            length_to_width_ratio = length / width
            width_to_length_ratio = width / length
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            if current_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

# ============================================================
# Cell 5: Ellipse Fitting Functions

def fit_and_draw_ellipses(image, optimized_contours, color):
    """Fits an ellipse to the optimized contours and draws it on the image."""
    if len(optimized_contours) >= 5:
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color, 2)
        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image


def check_contour_pixels(contour, image_shape, debug_mode_on):
    """
    Checks how many pixels in the contour fall under a slightly thickened ellipse.
    Returns number of pixels and ratio divided by total pixels on contour border.
    """
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


def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    """Check the quality of ellipse fit."""
    ellipse_goodness = [0, 0, 0]
    
    if len(contour) < 5:
        print("length of contour was 0")
        return 0
    
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness
    
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    axes_lengths = ellipse[1]
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

# ============================================================
# Cell 6: Frame Processing Functions

def process_frames(thresholded_image_strict, thresholded_image_medium, 
                    thresholded_image_relaxed, frame, gray_frame, 
                    darkest_point, debug_mode_on, render_cv_window):
    """Process multiple threshold versions and select best ellipse fit."""
    
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict]
    name_array = ["relaxed", "medium", "strict"]
    final_image = image_array[0]
    final_contours = []
    ellipse_reduced_contours = []
    goodness = 0
    best_array = 0 
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    # Iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            
            if debug_mode_on:
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                  
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:    " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1)
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1)
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1)
                cv2.putText(gray_copies[i-1], "final:      " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1)
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2)
        cv2.putText(test_frame, "Q     = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2)
        cv2.putText(test_frame, "D     = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2)

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)

    return final_rotated_rect


def process_frame(frame):
    """Finds the pupil in an individual frame and returns the center point."""
    
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, 
                                        thresholded_image_relaxed, frame, gray_frame, 
                                        darkest_point, False, False)
    
    return final_rotated_rect

# ============================================================
# Cell 7: Video Processing Function (MODIFIED)

def process_video(video_path, input_method):
    """Loads a video and finds the pupil in each frame."""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', fourcc, 30.0, (640, 480))

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    debug_mode_on = False
    temp_center = (0,0)

    while True:
        ret, frame = cap.read()
        if not ret:
            # === MODIFICATION START ===
            # If input is a video file, rewind to loop it
            if input_method == 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                # If it's a webcam or other error, break
                break
            # === MODIFICATION END ===

        frame = crop_to_aspect_ratio(frame)
        darkest_point = get_darkest_area(frame)

        if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
        
        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
        
        pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, 
                                            thresholded_image_relaxed, frame, gray_frame, 
                                            darkest_point, debug_mode_on, True)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d') and debug_mode_on == False:
            debug_mode_on = True
        elif key == ord('d') and debug_mode_on == True:
            debug_mode_on = False
            cv2.destroyAllWindows()
            
        if key == ord('q'):
            # out.release() # Removed: This is handled after the loop
            break   
            
        elif key == ord(' '):
            # === MODIFICATION START ===
            # Improved pause loop to also check for quit command
            quit_pressed = False
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break # Resume
                elif key == ord('q'):
                    quit_pressed = True
                    break # Quit
            if quit_pressed:
                break # Break main loop
            # === MODIFICATION END ===

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ============================================================
# Cell 8: File Selection and Main Execution

def select_video():
    """Prompts the user to select a video file if hardcoded path is not found."""
    root = tk.Tk()
    root.withdraw()
    video_path = 'C:/Google Drive/Eye Tracking/fulleyetest.mp4'
    
    if not os.path.exists(video_path):
        print("No file found at hardcoded path. Please select a video file.")
        video_path = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=[("Video Files", "*.mp4;*.avi")]
        )
        if not video_path:
            print("No file selected. Exiting.")
            return
            
    # Second parameter: 1 for video, 2 for webcam
    process_video(video_path, 1)


# ============================================================
# Cell 9: Run the program

if __name__ == "__main__":
    select_video()