import cv2
import numpy as np
import streamlit as st
import datetime

# Streamlit UI
st.title("Optical Flow Tracking App")
st.subheader("Tracking and detecting movement between two points")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Taking input frame (initial frame)
_, inp_img = cap.read()
inp_img = cv2.flip(inp_img, 1)
inp_img = cv2.blur(inp_img, (4,4))
gray_inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

# Initialize the points
old_pts = np.array([[350, 180], [350, 350]], dtype=np.float32).reshape(-1, 1, 2)
backup = old_pts.copy()
backup_img = gray_inp_img.copy()

# Output window for text
outp = np.zeros((480, 640, 3))

# Initialize position for text
ytest_pos = 40

# Streamlit frame rendering loop
while cap.isOpened():
    # Read a new frame
    _, new_inp_img = cap.read()
    new_inp_img = cv2.flip(new_inp_img, 1)
    new_inp_img = cv2.blur(new_inp_img, (4, 4))
    new_gray = cv2.cvtColor(new_inp_img, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_inp_img, new_gray, old_pts, None, maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08))

    # Boundaries for movement
    for i in range(4):
        if i % 2 == 0:  # X coordinates
            new_pts.ravel()[i] = max(min(new_pts.ravel()[i], 600), 20)
        else:  # Y coordinates
            new_pts.ravel()[i] = max(min(new_pts.ravel()[i], 350), 150)

    # Draw line between two points
    x, y = new_pts[0, :, :].ravel()
    a, b = new_pts[1, :, :].ravel()
    cv2.line(new_inp_img, (int(x), int(y)), (int(a), int(b)), (0, 0, 255), 15)

    # Check for boundary crossing
    if new_pts.ravel()[0] > 400 or new_pts.ravel()[2] > 400:
        if new_pts.ravel()[0] > 550 or new_pts.ravel()[2] > 550:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "gone at {}".format(datetime.datetime.now().strftime("%H:%M")), (10, ytest_pos),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

    elif new_pts.ravel()[0] < 200 or new_pts.ravel()[2] < 200:
        if new_pts.ravel()[0] < 50 or new_pts.ravel()[2] < 50:
            new_pts = backup.copy()
            new_inp_img = backup_img.copy()
            ytest_pos += 40
            cv2.putText(outp, "came at {}".format(datetime.datetime.now().strftime("%H:%M")), (10, ytest_pos),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))

    # Display the result in Streamlit
    st.image(new_inp_img, channels="BGR", caption="Optical Flow Tracking")

    # Update gray image and old points
    gray_inp_img = new_gray.copy()
    old_pts = new_pts.reshape(-1, 1, 2)

    # Exit condition for Streamlit (Stop button)
    if st.button("Stop"):
        break

# Release resources
cap.release()
