import ultralytics
import cv2
import numpy as np

# Load a model
model = ultralytics.YOLO("./weights/best.pt")

# Define a mapping from class indices to corner names
class_to_corner = {0: "bottom_left", 1: "bottom_right", 4: "top_left", 5: "top_right"}

# Run batched inference on a list of images
results = model(
    [
       "C:/Users/84913/Desktop/cccd.jpg"
    ]
)

# Process results list
for result in results:
    # Initialize a dictionary to store points
    points = {}

    # Iterate over detected objects
    for box in result.boxes:
        # Get the class index and coordinates
        class_index = int(box.cls.item())  # Convert tensor to int
        x1, y1, x2, y2 = box.xyxy[0]  # Assuming xyxy format

        # Calculate the center of the box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Map the class index to a corner name
        if class_index in class_to_corner:
            corner_name = class_to_corner[class_index]
            points[corner_name] = (center_x, center_y)

    # Check if all required points are detected
    if all(
        k in points for k in ["top_left", "top_right", "bottom_right", "bottom_left"]
    ):
        # Order the points
        rect = np.array(
            [
                points["top_left"],
                points["top_right"],
                points["bottom_right"],
                points["bottom_left"],
            ],
            dtype="float32",
        )
        # Define the destination points for the perspective transform
        width = int(np.linalg.norm(rect[0] - rect[1]))
        height = int(np.linalg.norm(rect[0] - rect[3]))
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )

        # Compute the perspective transform matrix and apply it
        image = cv2.imread(
            "C:/Users/84913/Desktop/cccd.jpg"
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        # Show the result
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not all corners detected.")
