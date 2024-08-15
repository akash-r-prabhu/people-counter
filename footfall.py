# %%
from ultralytics import YOLO
import cv2

# %%

model = YOLO("yolov8s.pt")
source = r"C:\Users\akash\Documents\tranquility\utils\test_1.mp4"
# source = r"C:\Users\akash\Downloads\Compressed\V000.seq_2\V000.seq"

# %%
outside_hall_count = 0
inside_hall_count = 0
total_entries = 0
total_exits = 0

line_coords = [(100, 200), (500, 200)]  


def is_above_line(point, line_coords):
    # returns if given point is above the line or not
    (x, y) = point
    (x1, y1), (x2, y2) = line_coords
    return y < y2



# %%
cap = cv2.VideoCapture(source)


# %%
drawing = False  
ix, iy = -1, -1  

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, annotated_frame,line_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        line_coords = [(ix, iy), (ix+200, iy)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            annotated_frame_copy = annotated_frame.copy()
            cv2.line(annotated_frame_copy, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('Frame', annotated_frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(annotated_frame, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow('Frame', annotated_frame)

# %%
person_positions = {}
first_frame=True
while True:
    success, frame = cap.read()


    if success:
        results=model.track(frame,persist=True,classes=[0])
    
        annotated_frame = results[0].plot()
        color = (0, 255, 0)
        thickness = 2
        line_length = 200
        x,y=ix,iy
        x1 = x - line_length // 2
        x2 = x + line_length // 2     
        cv2.line(annotated_frame, (x1, y), (x2, y), color, thickness)
        cv2.putText(annotated_frame, f"Outside hall: {outside_hall_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inside hall: {inside_hall_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Total entries: {total_entries}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Total exits: {total_exits}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if len(results[0].boxes) >0:
            
            for box in results[0].boxes:
                x,y,w,h=box.xywh.tolist()[0]
                x1,y1,x2,y2=int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)
                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,255,0),2)
                point = (int(x), int(y))
                above_line = is_above_line(point, line_coords)
                
                if not box.id:
                    continue
                id_person = box.id
                id_person = int(id_person)
                print(f"Person {id_person} is {'above' if above_line else 'below'} the line")
                if id_person not in person_positions:
                    person_positions[id_person] = "above" if above_line else "below"
                    if above_line:
                        outside_hall_count += 1
                    # else:
                    #     inside_hall_count += 1
                else:
                    if person_positions[id_person] == "above" and not above_line:
                        total_entries += 1
                        inside_hall_count += 1
                        outside_hall_count -= 1
                        person_positions[id_person] = "below"
                    elif person_positions[id_person] == "below" and above_line:
                        total_exits += 1
                        outside_hall_count += 1
                        inside_hall_count -= 1
                        person_positions[id_person] = "above"


        cv2.imshow("Frame", annotated_frame)
        if first_frame:
            cv2.setMouseCallback('Frame', draw_line)
            first_frame=False
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# %%



