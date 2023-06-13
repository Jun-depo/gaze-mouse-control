import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore

import time
import os
import cv2
import argparse
import sys
import argparse
import logging

from input_feeder import InputFeeder
from face_detection import Model_FaceDetection
from head_pose_estimation import Model_HeadPose
from facial_landmarks_detection import Model_FaceLandmark
from gaze_estimation import Model_GazeEstimation

from mouse_controller import MouseController

def main(args):

    logging.basicConfig(filename='app.log', level=logging.DEBUG)
    
    pre_model_loading = time.time()

    face_model_name = f"../intel/face-detection-adas-0001/{args.precision}/face-detection-adas-0001"
    
    head_pose_model_name = f"../intel/head-pose-estimation-adas-0001/{args.precision}/head-pose-estimation-adas-0001"

    face_landmark_model_name = f"../intel/landmarks-regression-retail-0009/{args.precision}/landmarks-regression-retail-0009"

    gaze_model_name = f"../intel/gaze-estimation-adas-0002/{args.precision}/gaze-estimation-adas-0002"

    device=args.device
    
    video_file="../bin/demo.mp4"
    
    threshold= args.threshold

    output_path = args.output_path
    
    input_feed = InputFeeder(input_type=args.input_type, input_file=video_file) # input_file=video_file

    input_feed.load_data()

    width = int(input_feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(input_feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video_frame_counts:", video_len)
    fps = int(input_feed.cap.get(cv2.CAP_PROP_FPS))

    # print("initial_w, initial_h", str(width) + ", " + str(height))
    out = cv2.VideoWriter(os.path.join(output_path, 'face_detect_output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height), True)

    face_model = Model_FaceDetection(model_name=face_model_name, device=args.device, threshold=threshold)

    face_model.load_model()

    head_pose_model = Model_HeadPose(model_name=head_pose_model_name, device=args.device)

    head_pose_model.load_model()

    face_landmark_model = Model_FaceLandmark(model_name=face_landmark_model_name, device=args.device)

    face_landmark_model.load_model()

    gaze_model = Model_GazeEstimation(model_name=gaze_model_name, device=args.device)
    
    gaze_model.load_model()

    end_model_loading = time.time()

    if args.get_perf_counts:
        print(f"model loading time: {end_model_loading-pre_model_loading:.4f}")

    while input_feed.cap.isOpened():    
        counter = 0

        input_time = []
        infer_time = []
        mouse_time = []

        for img in input_feed.next_batch(): 

            input_end = time.time()
            input_time.append(input_end - input_feed.start)

            if img is None:
                logging.info("No longer get image from video capture.")
                input_feed.close()
                break

            infer_start = time.time()

            boxes = face_model.predict(img)
            
            counter += 1                
        
            if len(boxes) > 0:

                for box in boxes:
                    # Each box [image_id, label, conf, x_min, y_min, x_max, y_max]
                    if len(box) == 0:
                        logging.debug("Error, No face_detection box detected from face_detection model")                        
                    
                    x_min, y_min, x_max, y_max = box[3:]

                    x_min, y_min, x_max, y_max = int(x_min*width), int(y_min*height), int(x_max*width), int(y_max*height)
                    

                    if args.display_boxes:
                        frame = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 125, 0), 2)
                        text1 = "id: " + str(box[1])
                        frame = cv2.putText(img, text1, (x_min+2, y_min-2), cv2.FONT_HERSHEY_SIMPLEX, \
                                            0.7, (90, 150, 0), 2, cv2.LINE_AA)

                    else:
                        frame = img

                    face_crop = img[y_min:y_max, x_min:x_max, :] 

                    head_pose_angles = head_pose_model.predict(face_crop)
                    #print("angle_y, angle_p, angle_r: ", head_pose_angles)

                    if len(head_pose_angles) == 0:
                        logging.debug("Error, No head_pose_angles detected from head_pose_estimation model")

                    landmarks = face_landmark_model.predict(face_crop)

                    if len(landmarks) == 0:
                        logging.debug("Error, No face landmarks detected from facial_landmarks_detection model") 
                    
                    x0, y0, x1, y1 = landmarks[:4]

                    x0 = int(x_min + (x_max-x_min)*x0)
                    x1 = int(x_min + (x_max-x_min)*x1)
                    y0 = int(y_min + (y_max-y_min)*y0)
                    y1 = int(y_min + (y_max-y_min)*y1)

                    if args.display_boxes:
                        frame = cv2.circle(frame, (x0,y0), radius=3, color=(0, 0, 255), thickness=-1)  # left side eye in img:m right eye
                        frame = cv2.circle(frame, (x1,y1), radius=3, color=(0, 255, 0), thickness=-1)  # right side eye in img:m left eye
                    
                    # left_eye (x0, y0), right_eye (x1,y1)
                    x_left = x0-35
                    y_left = y0-25
                    left_crop = img[y_left:y_left+50, x_left:x_left+70, :]

                    x_right = x1-35
                    y_right = y1-25
                    right_crop = img[y_right:y_right+50, x_right:x_right+70, :]

                    if args.display_boxes:
                        frame = cv2.rectangle(frame, (x_left, y_left), (x_left+70, y_left+50), (0, 0, 125), 2)
                        frame = cv2.rectangle(frame, (x_right, y_right), (x_right+70, y_right+50), (0, 0, 125), 2)

                    gaze_vector = gaze_model.predict(left_crop, right_crop, head_pose_angles)
                    # print("gaze_vector: ", gaze_vector)

                    if len(gaze_vector) == 0:
                        logging.debug("Error, No gaze_vector detected from gaze_estimation model")
                    
                    x, y = gaze_vector[:2]

                    infer_end = time.time() 

                    infer_time.append(infer_end - infer_start)

                    mouse_start = time.time()
                    mouse_controller = MouseController(precision="high", speed='fast')

                    mouse_controller.move(x, y)

                    mouse_end = time.time()
                    mouse_time.append(mouse_end - mouse_start)

                    # frame = cv2.circle(frame, (x, y), radius=3, color=(0, 255, 255), thickness=-1)

                    frame = cv2.resize(frame, (1920, 1080))
                    
                    # print("args.mouse_with_video", args.mouse_with_video)
                    if args.mouse_with_video:
                        cv2.imshow("Capturing", frame)

                    else:
                        continue

                    key_pressed = cv2.waitKey(10)
                    if key_pressed == 27: # escape key = 27           
                        break

                    out.write(frame)

        if args.get_perf_counts:
            print("img counts:", counter)

    if args.get_perf_counts:
        print(f"average input time per frame: {np.array(input_time).mean():.4f} ")
        print(f"average inference time per frame: {np.array(infer_time).mean(): .4f}")
        print(f"average mouse controlling time per frame: {np.array(mouse_time).mean():.4f}")
        print("totoal app running time: ", time.time()-pre_model_loading)

    out.release()
    input_feed.close
    cv2.destroyAllWindows()
        
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--precision', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--input_type', default="video")
    parser.add_argument('--output_path', default='../')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--get_perf_counts', default=False)
    parser.add_argument('--mouse_with_video', default=False)
    parser.add_argument('--display_boxes', default=False)
    
    args=parser.parse_args()

    main(args)
    