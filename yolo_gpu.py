import time
import os
import urllib
import queue
from threading import Thread

import cv2 as cv
import numpy as np
import pafy


def get_youtube(url):
    video = pafy.new(url)
    best = video.getbest(preftype='mp4')
    return best.url

CUR_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CUR_PATH, 'models/yolov3.weights')
CONFIG_PATH = os.path.join(CUR_PATH, 'models/yolov3.cfg')
CLASSES_PATH = os.path.join(CUR_PATH, 'coco.names')
# TARGET = 'https://www.youtube.com/watch?v='
TARGET = 'blackbox2.webm'
INPUT_PATH = os.path.join(CUR_PATH, TARGET) \
    if urllib.parse.urlparse(TARGET).netloc == '' else get_youtube(TARGET)
ASYNC_NUM = 0
SIZE = 608
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Load names of classes
CLASSES = None
if CLASSES_PATH:
    with open(CLASSES_PATH, 'rt') as f:
        CLASSES = f.read().rstrip('\n').split('\n')

# Load a network
net = cv.dnn.readNet(MODEL_PATH, CONFIG_PATH)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
out_names = net.getUnconnectedOutLayersNames()


def postprocess(frame, outs):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    def draw_pred(class_id, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if CLASSES:
            assert (class_id < len(CLASSES))
            label = '%s: %s' % (CLASSES[class_id], label)

        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(
            frame,
            (left, top - label_size[1]), (left + label_size[0], top + base_line),
            (255, 255, 255),
            cv.FILLED
        )
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layer_names = net.getLayerNames()
    last_layer_id = net.getLayerId(layer_names[-1])
    last_layer = net.getLayer(last_layer_id)

    class_ids = []
    confidences = []
    boxes = []
    # Network produces output blob with a shpae NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    if len(out_names) > 1:
        indices = []
        class_ids = np.array(class_ids)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(class_ids)
        for cl in unique_classes:
            class_indices = np.where(class_ids == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, CONF_THRESHOLD, NMS_THRESHOLD)
            nms_indices = nms_indices[:] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(class_ids))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_pred(class_ids[i], confidences[i], left, top, left + width, top + height)

# Process inputs
win_name = 'Deep learning object detection in OpenCV'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

def callback(pos):
    global CONF_THRESHOLD
    CONF_THRESHOLD = pos / 100.0

cv.createTrackbar('Confidence threshold, %', win_name, int(CONF_THRESHOLD * 100), 99, callback)

cap = cv.VideoCapture(INPUT_PATH)

class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.start_time = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.start_time = time.time()

    def get_FPS(self):
        return self.counter / (time.time() - self.start_time)


process = True

#
# Frames capturing thread
#
frames_queue = QueueFPS()
def frames_thread_body():
    global frames_queue, process

    while process:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frames_queue.put(frame)


#
# Frames processing thread
#
processed_frames_queue = queue.Queue()
predictions_queue = QueueFPS()
def processing_thread_body():
    global processed_frames_queue, predictions_queue, process

    while process:
        # Get a next frame
        frame = None
        try:
            frame = frames_queue.get_nowait()
            frames_queue.queue.clear()   # Skip the rest of frames
        except queue.Empty:
            pass

        if not frame is None:
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1/255.0, size=(SIZE, SIZE), swapRB=True, crop=False)
            processed_frames_queue.put(frame)

            # Run a model
            net.setInput(blob)
            outs = net.forward(out_names)
            predictions_queue.put(np.copy(outs))



frames_thread = Thread(target=frames_thread_body)
frames_thread.start()

processing_thread = Thread(target=processing_thread_body)
processing_thread.start()

#
# Postprocessing and rendering loop
#
while cv.waitKey(1) < 0:
    try:
        # Request prediction first because they put after frames
        outs = predictions_queue.get_nowait()
        frame = processed_frames_queue.get_nowait()

        postprocess(frame, outs)

        # Put efficiency information.
        if predictions_queue.counter > 1:
            label = 'Camera: %.2f FPS' % (frames_queue.get_FPS())
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            label = 'Network: %.2f FPS' % (predictions_queue.get_FPS())
            cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            label = 'Skipped frames: %d' % (frames_queue.counter - predictions_queue.counter)
            cv.putText(frame, label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow(win_name, frame)
    except queue.Empty:
        pass


process = False
frames_thread.join()
processing_thread.join()
