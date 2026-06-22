import os.path

import cv2
import numpy as np

MODEL_FILE = "best.onnx"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = {
    'Phone': 0.90,
    'Wrist Watch': 0.65,
    'Keys': 0.35,
    'Toothpaste': 0.50
}
NMS_THRESHOLD = 0.45

CLASS_NAMES = ['Keys', 'Phone', 'Toothpaste', 'Wrist Watch']

# overall app window
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

TOP_BAR_HEIGHT = 60

INCOMING_FEED_WIDTH = 320
SHELF_LAYOUT_WIDTH = 600

DASHBOARD_WIDTH = FRAME_WIDTH - INCOMING_FEED_WIDTH - SHELF_LAYOUT_WIDTH

INCOMING_FEED_START_X = 0
SHELF_LAYOUT_START_X = INCOMING_FEED_START_X + INCOMING_FEED_WIDTH
DASHBOARD_START_X = SHELF_LAYOUT_START_X + SHELF_LAYOUT_WIDTH

# shelf configurations
SHELF_ROWS = 4
SHELF_COLS = 4
NUM_SHELVES = SHELF_ROWS * SHELF_COLS
OBJECT_CAPACITY = 10

SHELF_GRID_OFFSET_X = SHELF_LAYOUT_START_X + 20
SHELF_GRID_OFFSET_Y = TOP_BAR_HEIGHT + 80

SHELF_CELL_WIDTH = (SHELF_LAYOUT_WIDTH - 40) // SHELF_COLS
SHELF_CELL_HEIGHT = (FRAME_HEIGHT - SHELF_GRID_OFFSET_Y - 20) // SHELF_ROWS
SHELF_PADDING_CELL = 8

# UI  COLORS

BG_DARK = (20, 20, 20)
BORDER_GRAY = (80, 80, 80)
TEXT_WHITE = (200, 200, 200)
ACCENT_BLUE = (255, 100, 200)
ACCENT_GREEN = (0, 200, 0)
WARNING_RED = (0, 0, 255)
FILL_EMPTY_COLOR = (40, 40, 40)
SHELF_BORDER_COLOR = (120, 120, 120)
SHELF_TEXT_COLOR = (180, 180, 180)

# camera feed
CAM_X = 20
CAM_Y = 80
CAM_WIDTH = 350
CAM_HEIGHT = 150

# shelf mappings
INVENTORY = [
    {"type": None, 'count': 0, 'max_capacity': OBJECT_CAPACITY}
    for _ in range(NUM_SHELVES)
]

CATEGORY_MAPPINGS = {
    "Phone": list(range(0, 4)),
    "Wrist Watch": list(range(4, 8)),
    "Keys": list(range(8, 12)),
    "Toothpaste": list(range(12, 16))
}

# cooldown for getting new objects
COOLDOWN_FRAMES = 90
cooldown_counter = 0

# fake opjects for testing
INVENTORY[0] = {'type': 'Phone', 'count': 4, 'max_capacity': 10}  # 40% full
INVENTORY[5] = {'type': 'Wrist Watch', 'count': 10, 'max_capacity': 10}  # 100% full (FULL)


def find_shelf_slot(obj_type):
    """Find the next available shelf slot"""
    if obj_type not in CATEGORY_MAPPINGS:
        return -1
    target_shelves = CATEGORY_MAPPINGS[obj_type]

    for i in target_shelves:
        shelf = INVENTORY[i]
        if shelf['type'] == obj_type and shelf['count'] < shelf['max_capacity']:
            return i
    for i in target_shelves:
        if INVENTORY[i]['type'] is None:
            return i
    return -1


def place_object(obj_type):
    """Allocates the object to inventory memory"""
    shelf_index = find_shelf_slot(obj_type)
    if shelf_index != -1:
        shelf = INVENTORY[shelf_index]
        if shelf['type'] is None:
            shelf['type'] = obj_type
        shelf['count'] += 1

        return True
    return False

def reset_inventory():
    """Wipes the database clean"""
    global cooldown_counter
    print("-----Resetting the Inventory State------")
    for i in range(NUM_SHELVES):
        INVENTORY[i]['type'] = None
        INVENTORY[i]['count'] = 0
    cooldown_counter = 0



def pre_process(incoming_image, net):
    "w.r.t YOLOv8 Model"
    (frame_h, frame_w) = incoming_image.shape[:2]
    r = min(INPUT_WIDTH / frame_w, INPUT_HEIGHT / frame_h)
    new_unpad_w, new_unpad_h = int(round(frame_w * r)), int(round(frame_h * r))
    dw, dh = (INPUT_WIDTH - new_unpad_w) // 2, (INPUT_HEIGHT - new_unpad_h) // 2

    resized_image = cv2.resize(incoming_image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    padded_image = cv2.copyMakeBorder(resized_image, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.dnn.blobFromImage(padded_image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs[0], r, dw, dh


def find_object(detection_zoneframe, net):
    (frame_h, frame_w) = detection_zoneframe.shape[:2]
    output_data, scale_ratio, pad_dw, pad_dh = pre_process(detection_zoneframe, net)
    output_data = output_data[0].T

    class_ids, confidences, boxes = [], [], []

    for row in output_data:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = float(scores[class_id])

        if class_id < len(CLASS_NAMES):
            item_name = CLASS_NAMES[class_id]
            req_confidence = CONFIDENCE_THRESHOLD.get(item_name, 0.5)

            if confidence >= req_confidence:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2))
                top = int((cy - h / 2))
                boxes.append([left, top, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, NMS_THRESHOLD)
    if len(indices) == 0: return None, None

    largest_area, best_detection = 0, None
    for i in indices.flatten():
        class_id = class_ids[i]
        if class_id < len(CLASS_NAMES):
            our_item_type = CLASS_NAMES[class_id]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            x1_orig = max(0, int(round((x - pad_dw) / scale_ratio)))
            y1_orig = max(0, int(round((y - pad_dh) / scale_ratio)))
            x2_orig = min(frame_w, x1_orig + int(round(w / scale_ratio)))
            # y2_orig = max(0, int(round((x - pad_dw) / scale_ratio)))
            y2_orig = min(frame_h, y1_orig + int(round(h / scale_ratio)))

            area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
            if area > largest_area:
                largest_area = area
                best_detection = (our_item_type, (x1_orig, y1_orig, x2_orig, y2_orig))

    if best_detection: return best_detection[0], best_detection[1]
    return None, None


def get_shelf_percentage(shelf_index):
    "indicates how full a specific shelf is"
    item = INVENTORY[shelf_index]
    if item['count'] == 0:
        return 0
    return (item['count'] / item['max_capacity']) * 100


def draw_rounded_rect(img, pt1, pt2, color, thickness, r):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


# ui drawing functions
def draw_top_bar(frame):
    "Draws the top navigation bar"

    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, TOP_BAR_HEIGHT), BG_DARK, -1)

    cv2.line(frame, (0, TOP_BAR_HEIGHT), (FRAME_WIDTH, TOP_BAR_HEIGHT), BORDER_GRAY, 1)
    cv2.putText(frame, "Vision Based Inventory Management System",
                (INCOMING_FEED_START_X + 20, TOP_BAR_HEIGHT // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_WHITE, 2)


def draw_main_sections_background(frame):
    "Draws vertical lines to seperate 2 sections"
    cv2.line(frame, (SHELF_LAYOUT_START_X, TOP_BAR_HEIGHT), (SHELF_LAYOUT_START_X, FRAME_HEIGHT), BORDER_GRAY, 1)

    cv2.line(frame, (DASHBOARD_START_X, TOP_BAR_HEIGHT), (DASHBOARD_START_X, FRAME_HEIGHT), BORDER_GRAY, 1)


def draw_incoming_feed(frame, webcam_frame):
    cv2.putText(frame, 'Incoming Feed', (INCOMING_FEED_START_X + 20, TOP_BAR_HEIGHT + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, TEXT_WHITE, 2)
    cam_x1 = INCOMING_FEED_START_X + 20
    cam_y1 = TOP_BAR_HEIGHT + 60
    cam_width = INCOMING_FEED_WIDTH - 40
    cam_height = 150

    if webcam_frame is not None:
        resized_cam = cv2.resize(webcam_frame, (cam_width, cam_height))
        frame[cam_y1:cam_y1 + cam_height, cam_x1:cam_x1 + cam_width] = resized_cam

    draw_rounded_rect(frame, (cam_x1, cam_y1), (cam_x1 + cam_width, cam_y1 + cam_height), BORDER_GRAY, 1, 10)

    status_y = cam_y1 + cam_height +30
    if cooldown_counter>0:
        cv2.putText(frame,"Processing Object...",(cam_x1,status_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,ACCENT_BLUE,1)
    else:
        cv2.putText(frame, "Waiting for Object...",(cam_x1, status_y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE, 1)



def draw_shelf_layout(frame):
    "draw the shelf and fill accordingly"
    section_x = SHELF_LAYOUT_START_X
    section_y = TOP_BAR_HEIGHT

    cv2.putText(frame, "Virtual Shelf Layout", (section_x + 40, section_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                TEXT_WHITE, 2)

    row_headers = {}
    for cat, shelves in CATEGORY_MAPPINGS.items():
        row = shelves[0] // SHELF_COLS
        row_headers[row] = cat

    SHELF_OFFSET_Y = 30

    for r in range(SHELF_ROWS):
        if r in row_headers:
            header_y = SHELF_GRID_OFFSET_Y + r * SHELF_CELL_HEIGHT + 20
            cv2.putText(frame, row_headers[r], (SHELF_GRID_OFFSET_X + SHELF_PADDING_CELL, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1)

        #         drawing the shelves
        for c in range(SHELF_COLS):
            index = r * SHELF_COLS + c
            shelf = INVENTORY[index]
            fill_pct = get_shelf_percentage(index)

            x1 = SHELF_GRID_OFFSET_X + c * SHELF_CELL_WIDTH
            y1 = SHELF_GRID_OFFSET_Y + r * SHELF_CELL_HEIGHT + SHELF_OFFSET_Y
            x2 = x1 + SHELF_CELL_WIDTH
            y2 = y1 + SHELF_CELL_HEIGHT - SHELF_OFFSET_Y

            draw_rounded_rect(frame, (x1 + SHELF_PADDING_CELL, y1 + SHELF_PADDING_CELL),
                              (x2 - SHELF_PADDING_CELL, y2 - SHELF_PADDING_CELL), SHELF_BORDER_COLOR, 1, 8)

            inner_x1 = x1 + SHELF_PADDING_CELL + 1
            inner_y1 = y1 + SHELF_PADDING_CELL + 1
            inner_x2 = x2 - SHELF_PADDING_CELL - 1
            inner_y2 = y2 - SHELF_PADDING_CELL - 1

            cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), FILL_EMPTY_COLOR, -1)
            #             if items presemt draw the empty bar
            if fill_pct > 0:
                fill_width = int((inner_x2 - inner_x1) * (fill_pct / 100.0))
                cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x1 + fill_width, inner_y2), (0, 0, 200), -1)

            cv2.putText(frame, f"S{index + 1}", (inner_x1 + 5, inner_y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, SHELF_TEXT_COLOR, 1)

            if shelf['type']:
                cv2.putText(frame, f"{int(fill_pct)}%", (inner_x1 + 5, inner_y1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_BLUE, 1)
            else:
                cv2.putText(frame, "EMPTY", (inner_x1 + 5, inner_y1 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, SHELF_TEXT_COLOR, 1)

            if fill_pct == 100:
                cv2.putText(frame, "Full", (inner_x2 - 30, inner_y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WARNING_RED, 1)


def draw_dashboard(frame):
    """Draws the summary guage and text tabe"""
    section_x, section_y = DASHBOARD_START_X, TOP_BAR_HEIGHT
    cv2.putText(frame, "Real-Time Dashboard", (section_x+ 20, section_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_WHITE, 2)

    y_current = section_y + 50
    current_fill = sum(item['count'] for item in INVENTORY)
    total_capacity = NUM_SHELVES * OBJECT_CAPACITY
    overall_fill_pct = (current_fill/total_capacity)*100 if total_capacity > 0 else 0

    gauge_center_x = section_x + DASHBOARD_WIDTH//2
    gauge_center_y = y_current + 60
    gauge_radius= 50

    cv2.putText(frame, 'Overall Fill', (gauge_center_x - 50, gauge_center_y- gauge_radius - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE,1)
    cv2.circle(frame,(gauge_center_x, gauge_center_y),gauge_radius, FILL_EMPTY_COLOR,-1)
    cv2.circle(frame,(gauge_center_x,gauge_center_y),gauge_radius, BORDER_GRAY,2)

    if overall_fill_pct>0:
        angle_end = int(270 + (overall_fill_pct/100.0)*360) % 360
        if angle_end == 270: angle_end = 269.9
        cv2.ellipse(frame, (gauge_center_x,gauge_center_y),(gauge_radius - 2, gauge_radius - 2),0,270,angle_end,ACCENT_BLUE,-1)

    cv2.putText(frame,f"{int(overall_fill_pct)}%",(gauge_center_x - 15, gauge_center_y + 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_WHITE,2)

    table_y_start = gauge_center_y + gauge_radius + 40
    # col_x= [section_x + 20, section_y + 100,section_x + 200]
    col1_x = section_x + 20
    col2_x = section_x + 80
    col3_x = section_x + 220


    cv2.putText(frame,"Id",(col1_x,table_y_start),cv2.FONT_HERSHEY_SIMPLEX,0.5,TEXT_WHITE,1)
    cv2.putText(frame, "Item", (col2_x, table_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1)
    cv2.putText(frame, "Fill %", (col3_x, table_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BORDER_GRAY, 1)
    cv2.line(frame, (section_x + 15, table_y_start + 5), (FRAME_WIDTH - 15, table_y_start + 5),BORDER_GRAY,1)
    y_current= table_y_start + 30
    for i in range(NUM_SHELVES):
        shelf = INVENTORY[i]
        fill_pct = get_shelf_percentage(i)
        item_next = shelf['type'] if shelf['type'] else "Empty"

        fill_display = f"{int(fill_pct)}%"

        cv2.putText(frame,f"S{i+1}",(col1_x,y_current),cv2.FONT_HERSHEY_SIMPLEX,0.4,TEXT_WHITE,1)
        cv2.putText(frame, item_next, (col2_x, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1)
        cv2.putText(frame, fill_display, (col3_x, y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ACCENT_GREEN if fill_pct>0 else TEXT_WHITE,1, 1)
        y_current +=20




def main():
    global cooldown_counter
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Put {MODEL_FILE} in project folder")
        return
    print('Loading my yolo model')
    net = cv2.dnn.readNet(MODEL_FILE)
    print('Model Ready')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    print("System started! Press q to Quit")

    webcam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            print('Error grabing frame')
            break

        webcam_frame = cv2.flip(webcam_frame, 1)

        main_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        main_canvas[:] = BG_DARK

        # draw the ui blueprint
        draw_top_bar(main_canvas)
        draw_main_sections_background(main_canvas)
        draw_incoming_feed(main_canvas, webcam_frame)
        draw_shelf_layout(main_canvas)
        draw_dashboard(main_canvas)

        if cooldown_counter>0:
            cooldown_counter-=1

        detected_type = None
        box = None

        # if cooldown_counter== 0:
        detected_type, box = find_object(webcam_frame,net)

        if detected_type and cooldown_counter == 0:
            print(f"Model Detected: {detected_type}")
            sucess = place_object(detected_type)
            cooldown_counter = COOLDOWN_FRAMES

            if not sucess:
                print(f'Shelf Full for {detected_type}')

        # detected_type, box = find_object(webcam_frame, net)

        if detected_type and box:
            cam_x1 = INCOMING_FEED_START_X + 20
            cam_y1 = TOP_BAR_HEIGHT + 60
            cam_width = INCOMING_FEED_WIDTH - 40
            cam_height = 150

            scale_x = cam_width / webcam_w
            scale_y = cam_height / webcam_h

            (x1, y1, x2, y2) = box
            ui_box_x1 = int(x1 * scale_x) + cam_x1
            ui_box_y1 = int(y1 * scale_y) + cam_y1
            ui_box_x2 = int(x2 * scale_x) + cam_x1
            ui_box_y2 = int(y2 * scale_y) + cam_y1

            cv2.rectangle(main_canvas, (ui_box_x1, ui_box_y1), (ui_box_x2, ui_box_y2), ACCENT_BLUE, 2)
            cv2.putText(main_canvas, detected_type, (ui_box_x1, ui_box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ACCENT_BLUE, 2)

        cv2.imshow("Vision - Based Inventory Management System", main_canvas)

        # show the canvas
        #
        # cam_x_end = CAM_X + CAM_WIDTH
        # cam_y_end = CAM_Y + CAM_HEIGHT
        #
        # main_canvas[CAM_Y:cam_y_end,CAM_X:cam_x_end] = resized_cam

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_inventory()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
