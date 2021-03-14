import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import threading


framework = 'tflite' #'tflite' # tf, tflite, trt
weights = './checkpoints/yolov4-320(20).tflite' #'./checkpoints/yolov4-416.tflite'
size = 320# resize images to
tiny = True  # 'yolo or yolo-tiny'
model = 'yolov4' # yolov3 or yolov4
iou = 0.45 # iou threshold
score = 0.65 # score threshold

def main(_argv):

    input_size = size

    vid = cv2.VideoCapture(1)

    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    frame_id = 0
    green = 0

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        

        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        height = input_size
        width= int(input_size/2)

        for y in range(0, height):
            for x in range(0,32):
                image_data[y,x] = (0,0,0)      
        
        for y in range(0, height):
            for x in range(160,320):
                image_data[y,x] = (0,0,0) 

        
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
 
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=5,
            max_total_size=5,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()] 

        

        y_avg1 = (pred_bbox[0][0][0][2]+pred_bbox[0][0][0][0])/2 

        y_avg2 = (pred_bbox[0][0][1][2]+pred_bbox[0][0][1][0])/2 

        y_avg3 = (pred_bbox[0][0][2][2]+pred_bbox[0][0][2][0])/2 

        y_avg4 = (pred_bbox[0][0][3][2]+pred_bbox[0][0][3][0])/2 

        y_avg5 = (pred_bbox[0][0][4][2]+pred_bbox[0][0][4][0])/2 
        if(green==1):
            if(scores[0][0]>0.30 and 0<y_avg1<=0.6 or 0<y_avg2<=0.6 or 0<y_avg3<=0.6 or 0<y_avg4<=0.6 or 0<y_avg5<=0.6):
                print("<<RED>>")

            elif(scores[0][0]>0.30 and 0<y_avg1>0.6 or 0<y_avg2>0.6 or 0<y_avg3>0.6 or 0<y_avg4>0.6 or 0<y_avg5>0.6):
                print("<<YELLO>>")
            else:
                print("<<BLACK>>")
        else:
            print("<<BLACK>>")

        image = utils.draw_bbox(frame, pred_bbox)
        #curr_time = time.time()
        #exec_time = curr_time - prev_time
        #result = np.asarray(image)
        #info = "time: %.2f ms" %(1000*exec_time)
        #print(info)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for y in range(0, 480):
            for x in range(63,65):
                result[y,x] = (0,0,255)  
        
        for y in range(0, 480):
            for x in range(319,321):
                result[y,x] = (0,0,255)  
        
        for y in range(359, 361):
            for x in range(64, 320):
                result[y,x] = (0,0,255)  

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('o'): green = 1
        if cv2.waitKey(1) & 0xFF == ord('p'): green = 0

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

