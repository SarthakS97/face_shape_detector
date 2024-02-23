from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import cv2 as cv
import model_train as dtc
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class FaceShapeDetector:
    def __init__(self):
        self._LBFModel = "data/lbfmodel.yaml"
        self._haarcascade = "data/lbpcascade_frontalface.xml"
        self._face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._landmark_detector = cv.face.createFacemarkLBF()
        self._landmark_detector.loadModel(self._LBFModel)

    def detect_face_shape(self, image):
        # Convert image to OpenCV format
        image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        # Resize the image to a specific width (e.g., 800 pixels)
        target_width = 800
        ratio = target_width / image_cv.shape[1]
        image_cv_resized = cv.resize(image_cv, (target_width, int(image_cv.shape[0] * ratio)))

        return self.detect_and_display(image_cv_resized, self._landmark_detector, "stillshot")

    def detect_and_display(self, frame, landmark_detector, method):
        # Convert image to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        # Rectangle face detection
        faces = self._face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5)

        landmark_points = []

        for (x, y, w, h) in faces:
            forehead = y
            forhead_mid = (x + (x+w)) // 2
            _, landmarks = landmark_detector.fit(frame_gray, faces)
            for landmark in landmarks:
                for x, y in landmark[0]:
                    landmark_points.append((int(x), int(y)))

            if len(landmark_points) >= 68:
                cheek_left = landmark_points[1]
                cheek_right = landmark_points[15]
                chin_left = landmark_points[6]
                chin_right = landmark_points[10]
                nose_left = landmark_points[3]
                nose_right = landmark_points[13]
                eye_brow_left = landmark_points[17]
                eye_brow_right = landmark_points[26]
                bottom_chin = landmark_points[8]
                cheek_bone_right_down_one = landmark_points[11]

                cheek_distance = cheek_right[0] - cheek_left[0]
                top_jaw_distance = nose_right[0] - nose_left[0]
                forehead_distance = eye_brow_right[0] - eye_brow_left[0]
                chin_distance = chin_right[0] - chin_left[0]
                head_lenghth = bottom_chin[1] - forehead

                jaw_width = top_jaw_distance
                jaw_right_to_down_one = cheek_bone_right_down_one[1] - \
                    nose_right[1]

                jaw_left_to_down_one = cheek_bone_right_down_one[0] - \
                    nose_left[0]

                jaw_angle = self._calculate_angle(jaw_width, jaw_right_to_down_one, jaw_left_to_down_one)

                return self.calculate_face_shape(
                    cheek_distance, top_jaw_distance, forehead_distance, chin_distance, head_lenghth, jaw_angle, method
                )

    def calculate_face_shape(self, cheek, jaw, forehead, chin, head_length, jaw_angle, method):
        cheek_ratio = cheek / head_length
        jaw_ratio = jaw / head_length
        forehead_ratio = forehead / head_length
        chin_ratio = chin / head_length
        head_ratio = head_length / cheek

        result = "Loading..."

        if (
            0.8 <= cheek_ratio <= 1.0 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio <= 1.25 and jaw_angle <= 50.0
        ):
            result = "Round Face"
        elif (
            0.5 <= cheek_ratio <= 0.8 and
            0.5 <= jaw_ratio <= 0.7 and
            0.5 <= forehead_ratio <= 0.7 and
            0.2 <= chin_ratio <= 0.4 and
            1.25 <= head_ratio <= 1.6 and jaw_angle > 50.0
        ):
            result = "Oval Face"
        elif (
            0.5 <= cheek_ratio <= 0.8 and
            0.5 <= jaw_ratio <= 0.8 and
            0.5 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio >= 1.30 and jaw_angle > 55
        ):
            result = "Rectangle Face"
        elif (
            0.7 <= cheek_ratio <= 0.99 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.99 and
            0.3 <= chin_ratio <= 0.5 and
            head_ratio <= 1.29 and jaw_angle < 55
        ):
            result = "Square Face"
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.5 <= forehead_ratio <= 0.7 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "Heart-Shaped Face"
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "Diamond Shaped Face"
        else:
            result = "Please adjust distance from camera"

        if method == "stillshot":
            descion_tree = dtc.PredictShape([cheek_ratio, jaw_ratio, forehead_ratio, chin_ratio, head_ratio, jaw_angle])
            classification = descion_tree.train_model()
            return classification[1][1][0].upper()

        return result

    def _calculate_angle(self, c, b, a):
        cosine_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        jaw_angle_degrees = np.degrees(np.arccos(cosine_angle))
        return jaw_angle_degrees

face_shape_detector = FaceShapeDetector()

@app.route('/upload', methods=['POST'])
def detect_face_shape():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    
    result = face_shape_detector.detect_face_shape(image)
    return jsonify({'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
