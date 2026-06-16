

import cv2
import numpy as np

# overall app window
FRAME_WIDTH = 1400
FRAME_HEIGHT = 800

# camera feed
CAM_X = 20
CAM_Y = 80
CAM_WIDTH = 350
CAM_HEIGHT = 150

def main():
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("System started! Press q to Quit")

    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            print('Error grabing frame')
            break

        webcam_frame = cv2.flip(webcam_frame,1)

        main_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH,3),dtype=np.uint8)

        main_canvas[:] = (20,20,20)

        resized_cam = cv2.resize(webcam_frame,(CAM_WIDTH,CAM_HEIGHT))

        cam_x_end = CAM_X + CAM_WIDTH
        cam_y_end = CAM_Y + CAM_HEIGHT

        main_canvas[CAM_Y:cam_y_end,CAM_X:cam_x_end] = resized_cam

        cv2.imshow("Vision Based inventory management System",main_canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()



