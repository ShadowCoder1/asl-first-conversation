/* First Conversation: only the MediaPipe pins (js/core/tracker.js reads them). */
export const MEDIAPIPE = {
  "version": "1.0.1",
  "delegate": "auto",
  "models": {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
  }
};
