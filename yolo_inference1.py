from ultralytics import YOLO
import os

# Load your trained YOLO model
model = YOLO('models/best.pt')  # Replace with the correct path to your model

# Path for saving annotations
annotations_dir = 'output_annotations'

# Create the directory to store annotations
if not os.path.exists(annotations_dir):
    os.makedirs(annotations_dir)

# Perform inference on the video with stream=True to avoid memory overflow
results = model.predict('input_videos/Bundesliga.mp4', save=True, stream=True)

# Set a minimum width and height threshold for boxes
MIN_BOX_WIDTH = 10  # Set a threshold for box width (pixels)
MIN_BOX_HEIGHT = 10  # Set a threshold for box height (pixels)

# Process and save results in YOLO format for each frame
for i, result in enumerate(results):  # Iterate over each result object from the generator
    print(f"Frame {i}: {len(result.boxes)} boxes detected")
    
    boxes = result.boxes
    if len(boxes) == 0:
        print(f"No boxes detected in frame {i}")
        continue  # Skip if no boxes detected in this frame
    
    annotations = []
    for box in boxes:
        # Print out the coordinates of each bounding box
        print(f"Box coordinates (xyxy): {box.xyxy}")  # Ensure xyxy is being used
        
        if box.xyxy is not None:
            # Clone the tensor to avoid modifying it directly
            coords = box.xyxy.clone()  # Clone the tensor
            
            # Extract the coordinates (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = coords[0].tolist()
            print(f"Extracted (xmin, ymin, xmax, ymax): {xmin}, {ymin}, {xmax}, {ymax}")
            
            width = xmax - xmin
            height = ymax - ymin
            print(f"Box width: {width}, Box height: {height}")
            
            # Check if the box meets the size threshold
            if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                print(f"Skipping box with too small dimensions: {width}x{height}")
                continue  # Skip boxes that are too small
            
            # Normalize coordinates to [0, 1] based on frame size
            frame_width = result.orig_shape[1]  # width of the frame
            frame_height = result.orig_shape[0]  # height of the frame
            
            # Normalize the coordinates (make sure it's not in-place modification)
            xmin /= frame_width
            ymin /= frame_height
            xmax /= frame_width
            ymax /= frame_height
            print(f"Normalized coordinates: {xmin}, {ymin}, {xmax}, {ymax}")
            
            # Get the class id and append the annotation
            class_id = int(box.cls)  # Class ID (e.g., player, referee)
            annotations.append(f"{class_id} {xmin} {ymin} {xmax} {ymax}")
        
        else:
            print(f"Box coordinates are None for this frame, skipping...")
    
    if annotations:
        # Save annotations to a text file (YOLO format)
        annotation_file = os.path.join(annotations_dir, f'frame_{i:04d}.txt')
        with open(annotation_file, 'w') as f:
            f.write('\n'.join(annotations))
        print(f"Annotations saved for frame {i:04d}")
    else:
        print(f"No valid annotations for frame {i:04d}")
















       

    


        





