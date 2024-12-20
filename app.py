import streamlit as st
import tempfile
from ultralytics import YOLO
import os
import cv2
import pandas as pd
import time

# Load the YOLO model
model = YOLO("best.pt")  # Replace 'best.pt' with the path to your trained model

# Function to process video and log errors
def process_video(video_path):
    try:
        # Temporary file for processed video
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_output.name

        # Run YOLO model
        st.write(f"Processing video: {video_path}")
        start_time = time.time()
        results = model.predict(source=video_path, conf=0.5, save=True, save_txt=False, save_conf=True)
        processing_time = time.time() - start_time
        st.write(f"Inference completed in {processing_time:.2f} seconds.")

        # Locate the processed video file
        default_save_dir = "runs/detect"
        latest_dir = sorted(
            [os.path.join(default_save_dir, d) for d in os.listdir(default_save_dir)],
            key=os.path.getctime,
            reverse=True
        )[0]

        processed_video_path = None
        for file in os.listdir(latest_dir):
            if file.endswith(".avi"):
                processed_video_path = os.path.join(latest_dir, file)
                break

        if not processed_video_path:
            st.error("Processed video not found in YOLO output.")
            return None

        # Error logging and FPS validation
        error_log = []
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        for frame_idx, frame_results in enumerate(results):
            frame_time = frame_idx / fps
            if frame_results.boxes is not None:
                detected_classes = [int(box[5]) for box in frame_results.boxes.data.tolist()]
                has_helmet = 0 in detected_classes
            else:
                has_helmet = False

            if not has_helmet:
                error_log.append({'Frame': frame_idx, 'Time (s)': round(frame_time, 2), 'Error': 'No helmet detected'})

        # Show error log
        if error_log:
            error_df = pd.DataFrame(error_log)
            st.write("Error Log:")
            st.dataframe(error_df)
        else:
            st.success("All helmets were detected successfully!")

        # Convert and return video
        if processed_video_path.endswith(".avi"):
            cap = cv2.VideoCapture(processed_video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()

            return output_path

        return processed_video_path

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app UI
st.title("Safety Helmet Detection")

# Video upload section
st.subheader("Upload a video for helmet detection:")
uploaded_file = st.file_uploader("Upload a video file (e.g., .mp4)", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_video_path, format="video/mp4")

    # Process video when the user clicks the "Process Video" button
    if st.button("Process Video"):
        with st.spinner("Processing... Please wait!"):
            processed_video_path = process_video(temp_video_path)

            if processed_video_path:
                st.success("Video processed successfully!")

                # Display and download processed video
                st.video(processed_video_path, format="video/mp4")
                with open(processed_video_path, "rb") as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
