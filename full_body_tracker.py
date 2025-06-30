import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

mp_pose = mp.solutions.pose

class FullBodyBiomarkerTracker(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            h, w, _ = image.shape
            lm = results.pose_landmarks.landmark

            def get_point(idx):
                pt = lm[idx]
                return int(pt.x * w), int(pt.y * h)

            # Landmarks
            pts = {
                "LShoulder": get_point(11),
                "RShoulder": get_point(12),
                "LElbow": get_point(13),
                "RElbow": get_point(14),
                "LWrist": get_point(15),
                "RWrist": get_point(16),
                "LPinky": get_point(17),
                "RPinky": get_point(18),
                "LHand": get_point(19),
                "RHand": get_point(20),
                "LHip": get_point(23),
                "RHip": get_point(24),
            }

            # Midpoints
            pts["MidShoulder"] = (
                (pts["LShoulder"][0] + pts["RShoulder"][0]) // 2,
                (pts["LShoulder"][1] + pts["RShoulder"][1]) // 2
            )
            pts["MidHip"] = (
                (pts["LHip"][0] + pts["RHip"][0]) // 2,
                (pts["LHip"][1] + pts["RHip"][1]) // 2
            )

            # Draw points
            for name, (x, y) in pts.items():
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Connect joints (arms, hands)
            for side in ["L", "R"]:
                cv2.line(image, pts[f"{side}Shoulder"], pts[f"{side}Elbow"], (255, 0, 0), 2)
                cv2.line(image, pts[f"{side}Elbow"], pts[f"{side}Wrist"], (0, 255, 255), 2)
                cv2.line(image, pts[f"{side}Wrist"], pts[f"{side}Hand"], (0, 128, 255), 2)
                cv2.line(image, pts[f"{side}Hand"], pts[f"{side}Pinky"], (128, 0, 255), 2)

            # Connect torso
            cv2.line(image, pts["LShoulder"], pts["RShoulder"], (255, 255, 255), 2)
            cv2.line(image, pts["LHip"], pts["RHip"], (255, 255, 255), 2)
            cv2.line(image, pts["LShoulder"], pts["LHip"], (100, 255, 100), 2)
            cv2.line(image, pts["RShoulder"], pts["RHip"], (100, 255, 100), 2)
            cv2.line(image, pts["MidShoulder"], pts["MidHip"], (0, 0, 255), 2)

            # === BIOMARKERS ===
            def angle(a, b, c):
                a, b, c = np.array(a), np.array(b), np.array(c)
                ba = a - b
                bc = c - b
                cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

            biomarkers = {
                "L Elbow Angle": angle(pts["LShoulder"], pts["LElbow"], pts["LWrist"]),
                "R Elbow Angle": angle(pts["RShoulder"], pts["RElbow"], pts["RWrist"]),
                "Shoulder-Hip Separation": angle(pts["LShoulder"], pts["MidShoulder"], pts["MidHip"]),
                "Torso Tilt (deg)": np.degrees(np.arctan2(
                    pts["MidShoulder"][1] - pts["MidHip"][1],
                    pts["MidShoulder"][0] - pts["MidHip"][0]
                ))
            }

            # Overlay biomarkers
            y_offset = 30
            for label, val in biomarkers.items():
                cv2.putText(image, f"{label}: {val:.1f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                y_offset += 25

        return image

# Streamlit UI
st.title("Full-Body Biomechanical Tracker")
st.caption("Live tracking of shoulder, elbow, wrist, hand, hip, and spine with elbow angles, torso tilt, and separation.")
webrtc_streamer(key="full-biomarkers", video_transformer_factory=FullBodyBiomarkerTracker)
