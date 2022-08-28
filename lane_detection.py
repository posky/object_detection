import os
import time

import numpy as np
import cv2 as cv

from lane_utils import initialize_trackbars, undistort, thresholding
from lane_utils import val_trackbars, perspective_warp, sliding_window
from lane_utils import get_curve, draw_lanes, draw_lines, perspective_warp_one_point
from lane_utils import stack_images


# path
CUR_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(CUR_PATH, 'models/yolov3.cfg')
WEIGHTS_PATH = os.path.join(CUR_PATH, 'models/yolov3.weights')
# VIDEO_PATH = os.path.join(CUR_PATH, 'input/project_video.mp4')
VIDEO_PATH = os.path.join(CUR_PATH, 'input/blackbox3.webm')
OUTPUT_PATH = os.path.join(CUR_PATH, 'output/')
CLASSES_PATH = os.path.join(CUR_PATH, 'models/coco.names')

FONT = cv.FONT_HERSHEY_PLAIN
CLASSES = []
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
VEHICLES = ['car', 'motorbike', 'bus', 'truck']

net = cv.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
with open(CLASSES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

frame_id = 0
init_trackbar_vals = [42, 63, 14, 87]      # wT, hT, wB, hB

cap = cv.VideoCapture(VIDEO_PATH)

count = 0
NO_OF_ARRAY_VALUES = 10
array_counter = 0
array_curve = np.zeros([NO_OF_ARRAY_VALUES])
my_vals = []
initialize_trackbars(init_trackbar_vals)


start_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        print('[i] ==> Done processing!!!')
        cv.waitKey(1000)
        break

    img = cv.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), None)
    img_final = img.copy()
    img_canny = img.copy()

    img_undis = undistort(img)
    img_thres, img_canny, img_color = thresholding(img_undis)
    src = val_trackbars()
    img_warp = perspective_warp(img_thres, dst_size=(FRAME_WIDTH, FRAME_HEIGHT), src=src)
    img_sliding, curves, lanes, ploty = sliding_window(img_warp, draw_windows=True)

    try:
        curverad = get_curve(img_final, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        img_final = draw_lanes(img, curves[0], curves[1], FRAME_WIDTH, FRAME_HEIGHT, src=src)

        # Average
        current_curve = lane_curve // 50
        if int(np.sum(array_curve)) == 0:
            average_curve = current_curve
        else:
            average_curve = np.sum(array_curve) // array_curve.shape[0]
        if abs(average_curve - current_curve) > 200:
            array_curve[array_counter] = average_curve
        else:
            array_curve[array_counter] = current_curve
        array_counter += 1
        if array_counter >= NO_OF_ARRAY_VALUES:
            array_counter = 0
        cv.putText(
            img_final,
            str(int(average_curve)),
            (FRAME_WIDTH // 2 - 70, 70),
            cv.FONT_HERSHEY_DUPLEX,
            1.75,
            (0, 0, 255),
            2,
            cv.LINE_AA
        )
    except:
        lane_curve = 0

    img_final = draw_lines(img_final, lane_curve)

    # Object detection
    success, frame = cap.read()

    frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), None)
    frame_id += 1
    height, width, channels = frame.shape
    # Detect image
    blob = cv.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                # Name of the object
                class_ids.append(class_id)
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    detected = False
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f'{CLASSES[class_ids[i]]}: {confidences[i] * 100:.2f}%'
            color = colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label, (x, y + 10), FONT, 2, color, 2)
            if CLASSES[class_ids[i]] in VEHICLES:
                center_x, center_y = int(x + w / 2), int(y + h / 2)
                warped_x, warped_y = perspective_warp_one_point(
                    img, (center_x, center_y), dst_size=(FRAME_WIDTH, FRAME_HEIGHT), src=src
                )

                if not isinstance(curves[0], int) and \
                        0 < warped_y < FRAME_HEIGHT and \
                        0 < curves[0][warped_y] < FRAME_WIDTH and \
                        0 < curves[1][warped_y] < FRAME_WIDTH and \
                        curves[0][warped_y] < warped_x < curves[1][warped_y]:
                    detected = True
                    cv.putText(frame, 'Warning!', (center_x, center_y), FONT, 2, color, 2)
                    cv.putText(frame, 'Detected on lane', (30, 60), FONT, 2, (0, 0, 0), 1)

    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    cv.putText(frame, 'FPS:' + str(fps), (10, 30), FONT, 2, (0, 0, 0), 1)
    img_blank = np.zeros_like(img)

    img_stacked = stack_images(
        0.7, ([img_undis, frame], [img_color, img_canny], [img_warp, img_sliding])
    )

    cv.imshow('Image', frame)
    cv.imshow('PipeLine', img_stacked)
    cv.imshow('Result', img_final)

    if detected:
        key = cv.waitKey(0)
        if key == ord('q'):
            break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print('==> All done!')
print('***************************************************************')
