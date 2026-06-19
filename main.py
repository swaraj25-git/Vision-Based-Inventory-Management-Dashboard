

import cv2
import numpy as np

# overall app window
FRAME_WIDTH = 1400
FRAME_HEIGHT = 800

TOP_BAR_HEIGHT = 60

INCOMING_FEED_WIDTH = 350
SHELF_LAYOUT_WIDTH = 650

DASHBOARD_WIDTH = FRAME_WIDTH - INCOMING_FEED_WIDTH - SHELF_LAYOUT_WIDTH

INCOMING_FEED_START_X = 0
SHELF_LAYOUT_START_X = INCOMING_FEED_START_X + INCOMING_FEED_WIDTH
DASHBOARD_START_X = SHELF_LAYOUT_START_X + SHELF_LAYOUT_WIDTH

# shelf configurations
SHELF_ROWS = 4
SHELF_COLS = 4
NUM_SHELVES = SHELF_ROWS*SHELF_COLS
OBJECT_CAPACITY = 10

SHELF_GRID_OFFSET_X = SHELF_LAYOUT_START_X +20
SHELF_GRID_OFFSET_Y = TOP_BAR_HEIGHT + 80
SHELF_CELL_WIDTH = (SHELF_LAYOUT_WIDTH-40)
SHELF_CELL_HEIGHT = (FRAME_HEIGHT - SHELF_GRID_OFFSET_Y- 20)
SHELF_PADDING_CELL = 8



# UI  COLORS

BG_DARK = (20,20,20)
BORDER_GRAY = (80,80,80)
TEXT_WHITE = (200,200,200)
ACCENT_BLUE = (255,100,200)
WARNING_RED = (0,0,255)
FILL_EMPTY_COLOR = (40,40,40)
SHELF_BORDER_COLOR = (120,120,120)
SHELF_TEXT_COLOR = (180,180,180)

# camera feed
CAM_X = 20
CAM_Y = 80
CAM_WIDTH = 350
CAM_HEIGHT = 150


# shelf mappings
INVENTORY = [
    {"type":None, 'count':0,'max_capacity':OBJECT_CAPACITY}
    for _ in range(NUM_SHELVES)
]

CATEGORY_MAPPINGS = {
    "Phone":list(range(0,4)),
    "Wrist Watch":list(range(4,8)),
    "Keys":list(range(8,12)),
    "Toothpaste":list(range(12,16))
}

# fake opjects for testing
INVENTORY[0] = {'type': 'Phone', 'count': 4, 'max_capacity': 10}       # 40% full
INVENTORY[5] = {'type': 'Wrist watch', 'count': 10, 'max_capacity': 10} # 100% full (FULL)


def get_shelf_percentage(shelf_index):
    "indicates how full a specific shelf is"
    item = INVENTORY[shelf_index]
    if item['count'] == 0:
        return 0
    return (item['count']/item['max_capacity']) * 100

def draw_rounded_rect(img,pt1,pt2,color,thickness,r):
    x1,y1 = pt1
    x2,y2 = pt2
    cv2.ellipse(img,(x1 + r, y1 + r ),(r,r),180,0,90,color,thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.line(img,(x1 + r, y1),(x2 - r,y1),color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

# ui drawing functions
def draw_top_bar(frame):
    "Draws the top navigation bar"

    cv2.rectangle(frame,(0,0),(FRAME_WIDTH,TOP_BAR_HEIGHT),BG_DARK,-1)

    cv2.line(frame, (0,TOP_BAR_HEIGHT),(FRAME_WIDTH,TOP_BAR_HEIGHT),BORDER_GRAY,1)
    cv2.putText(frame,"Vision Based Inventory Management System",(INCOMING_FEED_START_X+20, TOP_BAR_HEIGHT//2 +8),cv2.FONT_HERSHEY_SIMPLEX,0.8,TEXT_WHITE,2)

def draw_main_sections_background(frame):
    "Draws vertical lines to seperate 2 sections"
    cv2.line(frame,(SHELF_LAYOUT_START_X,TOP_BAR_HEIGHT),(SHELF_LAYOUT_START_X,FRAME_HEIGHT),BORDER_GRAY,1)

    cv2.line(frame,(DASHBOARD_START_X,TOP_BAR_HEIGHT),(DASHBOARD_START_X, FRAME_HEIGHT),BORDER_GRAY,1)

def draw_incoming_feed(frame, webcam_frame):
    cv2.putText(frame,'Incoming Feed',(INCOMING_FEED_START_X + 20, TOP_BAR_HEIGHT+ 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,TEXT_WHITE,2)
    cam_x1 = INCOMING_FEED_START_X + 20
    cam_y1 = TOP_BAR_HEIGHT + 60
    cam_width = INCOMING_FEED_WIDTH - 40
    cam_height = 150

    if webcam_frame is not None:
        resized_cam = cv2.resize(webcam_frame,(cam_width,cam_height))
        frame[cam_y1:cam_y1+cam_height, cam_x1:cam_x1+cam_width] = resized_cam

    draw_rounded_rect(frame, (cam_x1,cam_y1),(cam_x1+cam_width,cam_y1+cam_height),BORDER_GRAY,1,10)

def draw_shelf_layout(frame):
    "draw the shelf and fill accordingly"
    section_x = SHELF_LAYOUT_START_X
    section_y = TOP_BAR_HEIGHT

    cv2.putText(frame, "Virtual Shelf Layout", (section_x+20, section_y+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,TEXT_WHITE,2)

    row_headers = {}
    for cat, shelves in CATEGORY_MAPPINGS.items():
        row = shelves[0] // SHELF_COLS
        row_headers[row] = cat


    SHELF_OFFSET_Y = 30

    for r in range(SHELF_ROWS):
        if r in row_headers:
            header_y = SHELF_GRID_OFFSET_Y + r * SHELF_CELL_HEIGHT + 20
            cv2.putText(frame, row_headers[r], (SHELF_GRID_OFFSET_X + SHELF_PADDING_CELL, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE,1)

#         drawing the shelves
        for c in range(SHELF_COLS):
            index = r*SHELF_COLS + c
            shelf = INVENTORY[index]
            fill_pct = get_shelf_percentage(index)

            x1 = SHELF_GRID_OFFSET_X + c * SHELF_CELL_WIDTH
            y1 = SHELF_GRID_OFFSET_Y + r * SHELF_CELL_HEIGHT + SHELF_OFFSET_Y
            x2 = x1 + SHELF_CELL_HEIGHT
            y2= y1 + SHELF_CELL_HEIGHT - SHELF_OFFSET_Y

            draw_rounded_rect(frame, (x1 + SHELF_PADDING_CELL, y1 + SHELF_PADDING_CELL),
                              (x2 -  SHELF_PADDING_CELL, y2 - SHELF_PADDING_CELL),SHELF_BORDER_COLOR,1,8)

            inner_x1 = x1 + SHELF_PADDING_CELL + 1
            inner_y1 = y1 + SHELF_PADDING_CELL + 1
            inner_x2 = x2 - SHELF_PADDING_CELL - 1
            inner_y2 = y2 - SHELF_PADDING_CELL - 1

            cv2.rectangle(frame, (inner_x1,inner_y1),(inner_x2,inner_y2),FILL_EMPTY_COLOR,-1)
#             if items presemt draw the empty bar
            if fill_pct>0:
                fill_width = int((inner_x2-inner_x1)* (fill_pct/100.0))
                cv2.rectangle(frame,(inner_x1,inner_y1),(inner_x1+fill_width,inner_y2),(0,0,200),-1)

            cv2.putText(frame, f"S{index +1}",(inner_x1 + 5, inner_y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,SHELF_TEXT_COLOR,1)

            if shelf['type']:
                cv2.putText(frame, f"{int(fill_pct)}%",(inner_x1 + 5, inner_y1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,ACCENT_BLUE,1)
            else:
                cv2.putText(frame, "EMPTY",(inner_x1 + 5, inner_y1 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, SHELF_TEXT_COLOR,1)

            if fill_pct== 100:
                cv2.putText(frame, "Full", (inner_x2 - 30, inner_y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,WARNING_RED,1)

def main():
    cap = cv2.VideoCapture(0)
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


        main_canvas[:] = BG_DARK

        # draw the ui blueprint
        draw_top_bar(main_canvas)
        draw_main_sections_background(main_canvas)
        draw_incoming_feed(main_canvas, webcam_frame)

        draw_shelf_layout(main_canvas)
        cv2.imshow("Vision - Based Inventory Management System",main_canvas)



        # show the canvas
        #
        # cam_x_end = CAM_X + CAM_WIDTH
        # cam_y_end = CAM_Y + CAM_HEIGHT
        #
        # main_canvas[CAM_Y:cam_y_end,CAM_X:cam_x_end] = resized_cam


        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()



