# Emotion-Detection
Real-Time Emotion and Sleep Detection
This project is a Real-Time Emotion and Sleep Detection System that uses a webcam feed to analyze facial expressions and detect drowsiness based on eye closure duration. It leverages DeepFace for emotion recognition and dlib for facial landmark detection. The project is designed to enhance engagement and monitor alertness in real-time, making it useful for applications such as e-learning, driver monitoring, and more.

Features
1.Emotion Detection:

Recognizes the dominant emotion of the detected face using the DeepFace library.
Displays the emotion on the screen in real-time.
2.Sleep Detection:

Monitors eye closure to detect signs of drowsiness.
Alerts the user if the eyes remain closed for more than 5 seconds.
3.Real-Time Processing:

Processes live webcam video feed with minimal latency.
4.User-Friendly Interface:

Built with Streamlit for easy deployment and interaction

Libraries
Ensure the following Python libraries are installed:

deepface
opencv-python
numpy
dlib
streamlit
Additional Files
Download the shape predictor file (shape_predictor_68_face_landmarks.dat) from dlib's GitHub repository and extract it. Place the file in the appropriate directory.
