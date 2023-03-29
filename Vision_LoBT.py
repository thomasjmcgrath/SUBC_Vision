import cv2
import numpy as np


def main():
    # Replace 'input_video.mp4' with the path to your video file or use '0' for webcam feed
    video_path = 'input_video.mp4' 
    capture = cv2.VideoCapture(video_path)

    # Set playback speed (in milliseconds, 30 ms = approximately 33.3 fps)
    playback_speed = 1  # Adjust this value to change the playback speed (1 is default)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow('Processed Frame', processed_frame)


        # Press 'q' to exit the loop
        if cv2.waitKey(playback_speed) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(frame)

    # Use Otsu's method to automatically detect bimodal differences in hue and value
    ret_h,th_h = cv2.threshold(h,0,60,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_s,th_s = cv2.threshold(s,127,200,cv2.THRESH_BINARY) # make sure to binarize one of them without otsu's
    ret_v,th_v = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Threshold the products of those
    ret_m,mask = cv2.threshold(th_h*th_s*th_v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply a Gaussian blur to reduce noise
    blurred_mask = cv2.medianBlur(mask, 3)
    blurred_mask = cv2.Sobel(blurred_mask, cv2.CV_8U, 1, 0, ksize=5, scale=1)
    blurred_mask = cv2.GaussianBlur(blurred_mask, (5, 5), 1)

    # Find lines using the HoughLinesP function
    lines = cv2.HoughLinesP(blurred_mask, 2, np.pi / 60, threshold=1, minLineLength=20, maxLineGap=700)

    # Initialize two lists for circle points of different colors
    magenta_points = []
    green_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if the line is vertical (allowing for a small margin of error)
            if abs(x2 - x1) < 3:
                # Add points to the respective lists
                magenta_points.append((x1, y1))
                green_points.append((x2, y2))

                # Draw circles at the endpoints
                cv2.circle(frame, (x1, y1), 3, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 3, (0, 255, 0), -1)

    # Fit a line of best fit if there are enough points
    if len(magenta_points) > 1 and len(green_points) > 1:
        magenta_points = np.array(magenta_points)
        green_points = np.array(green_points)

        # Combine both lists of points
        combined_points = np.vstack((magenta_points, green_points))

        # Fit a single line using NumPy's polyfit function
        combined_fit = np.polyfit(combined_points[:, 0], combined_points[:, 1], 1)

        # Calculate the angle of the line of best fit in degrees
        combined_angle = np.arctan(combined_fit[0]) * 180 / np.pi

        # Set a threshold for deviation from the vertical axis (in degrees)
        angle_threshold = 45  # you can adjust this value as needed

        # Draw the line of best fit on the frame if its angle is within the threshold
        if abs(90 - abs(combined_angle)) <= angle_threshold:
            cv2.line(frame, (0, int(combined_fit[1])), (frame.shape[1], int(combined_fit[0] * frame.shape[1] + combined_fit[1])), (66, 132, 245), 2)

    return frame
if __name__ == "__main__":
    main()