import numpy as np
import cv2 as cv
import matplotlib.image as mpimg
import pickle

import timeit


def nothing(x):
    pass

def initialize_trackbars(init_trackbar_vals):
    cv.namedWindow('Trackbars')
    cv.resizeWindow('Trackbars', 360, 240)
    cv.createTrackbar('Width Top', 'Trackbars', init_trackbar_vals[0], 50, nothing)
    cv.createTrackbar('Height Top', 'Trackbars', init_trackbar_vals[1], 100, nothing)
    cv.createTrackbar('Width Bottom', 'Trackbars', init_trackbar_vals[2], 50, nothing)
    cv.createTrackbar('Height Bottom', 'Trackbars', init_trackbar_vals[3], 100, nothing)

def undistort(img, cal_dir='models/calibration_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv.undistort(img, mtx, dist, None, mtx)
    return dst

def color_filter(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])
    masked_white = cv.inRange(hsv, lower_white, upper_white)
    masked_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    combined_image = cv.bitwise_or(masked_white, masked_yellow)
    return combined_image

def thresholding(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5))
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv.Canny(img_blur, 50, 100)
    img_dial = cv.dilate(img_canny, kernel, iterations=1)
    img_erode = cv.erode(img_dial, kernel, iterations=1)

    img_color = color_filter(img)
    combined_image = cv.bitwise_or(img_color, img_erode)

    return combined_image, img_canny, img_color

def val_trackbars():
    width_top = cv.getTrackbarPos('Width Top', 'Trackbars')
    height_top = cv.getTrackbarPos('Height Top', 'Trackbars')
    width_bottom = cv.getTrackbarPos('Width Bottom', 'Trackbars')
    height_bottom = cv.getTrackbarPos('Height Bottom', 'Trackbars')

    src = np.float32([
        (width_top / 100, height_top / 100),
        (1 - (width_top / 100), height_top / 100),
        (width_bottom / 100, height_bottom / 100),
        (1 - (width_bottom / 100), height_bottom / 100)
    ])

    return src

def perspective_warp(
        img,
        dst_size=(1280, 720),
        src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
        dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV WwarpPerspective()
    warped = cv.warpPerspective(img, M, dst_size)

    return warped

def inv_perspective_warp(
        img,
        dst_size=(1280, 720),
        src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
        dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    # for detination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, bu close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv.warpPerspective(img, M, dst_size)
    return warped

def draw_points(img, src):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    for x in range(0, 4):
        cv.circle(img, (int(src[x][0]), int(src[x][1])), 15, (0, 0, 255), cv.FILLED)

    return img


def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c

    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(
                out_img,
                (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                (100, 255, 255),
                1
            )
            cv.rectangle(
                out_img,
                (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                (100, 255, 255),
                1
            )
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size and rightx.size:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    else:
        return img, (0, 0), (0, 0), 0

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 1 / img.shape[0]   # meters per pixel in y dimension
    xm_per_pix = 0.1 / img.shape[0] # meters per pixel in x dimension

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters

    return (l_fit_x_int, r_fit_x_int, center)

def draw_lanes(img, left_fit, right_fit, frame_width, frame_height, src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv.fillPoly(color_img, np.int_(points), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_img, (frame_width, frame_height), dst=src)
    inv_perspective = cv.addWeighted(img, 0.5, inv_perspective, 0.7, 0)

    return inv_perspective

def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height =  img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor

    return ver

def draw_lines(img, lane_curve):
    my_width = img.shape[1]
    my_height = img.shape[0]
    print(my_width, my_height)
    for x in range(-30, 30):
        w = my_width // 20
        cv.line(img, (w * x + int(lane_curve // 100), my_height - 30), (w * x + int(lane_curve // 100), my_height), (0, 0, 255), 2)
    cv.line(img, (int(lane_curve // 100) + my_width // 2, my_height - 30), (int(lane_curve // 100) + my_width // 2, my_height), (0, 255, 0), 3)
    cv.line(img, (my_width // 2, my_height - 50), (my_width // 2, my_height), (0, 255, 255), 2)

    return img
