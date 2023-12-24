import os
# comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import sqlite3
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import sys

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('output', './Crowd_Density/yolov4-deepsort/outputs', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
# flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
# flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

# conn = sqlite3.connect('Database_3.db')
# cur = conn.cursor()


# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1920,
#     capture_height=1080,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc sensor-id=%d !"
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             sensor_id,
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )
# pipeline = " ! ".join(["v4l2src device=/dev/video0",
#                        "video/x-raw, width=640, height=480, framerate=30/1",
#                        "videoconvert",
#                        "video/x-raw, format=(string)BGR",
#                        "appsink"
#                        ])
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    camera_id = "/dev/video0"
    # print(gstreamer_pipeline(flip_method=0))
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    # video_path = './data/video/Test_Trim.mp4'
    # vid  = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    vid = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # try:
    #     vid = cv2.VideoCapture(int(video_path))
    # except:
    #     vid = cv2.VideoCapture(video_path)
    # output = './Crowd_Density/yolov4-deepsort/outputs/output.avi'
    out = None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    # out = cv2.VideoWriter("output'.avi", codec, fps, (width, height))
    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        count = 2
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = count+len(names)
        # cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
        #             (0, 255, 0), 2)
        print("Number of people : {}".format(count))
        if count>5:
            print("WARNING CROWD LIMIT REACHED")
            cv2.putText(frame, "WARNING CROWD LIMIT REACHED" , (5, 350), 0, 0.75, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "CROWD UNDER CONTROL" , (5, 350), 0, 0.75, (255, 255, 255), 2)
        # cur.execute("INSERT INTO People_number(id,Number) VALUES (?,?)",(frame_num,count,))
        # cur.execute("INSERT INTO People_number VALUES (1)")
        # conn.commit()

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        frame= utils.draw_bbox(frame, pred_bbox)
        cv2.putText(frame, "No of People:"+str(count), (240, 20), 0, 0.75,(255, 255, 255), 2)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)
        print(0)
        # out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    # cap.release()
    # out.release()
    vid.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        app.run(main)
        # conn.commit()
        # print("Records created successfully")
        # conn.close()
    except SystemExit:
        pass

