
# Traffic Sign Recognition

This project implements a real-time driver assistance system using traffic sign recognition, lane detection, and collision detection.

## Directory Structure

```
TrafficVisionAI/
├── data/
│   ├── GTSRB/
│   │   ├── train/
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   └── ...
│   │   ├── Meta.csv
│   │   ├── Test.csv
│   │   └── Train.csv
├── models/
│   └── traffic_sign_model.h5
├── src/
│   ├── __init__.py
│   ├── traffic_sign_recognition.py
│   ├── lane_detection.py
│   ├── collision_detection.py
│   ├── main.py
│   ├── train_model.py
├── utils/
│   ├── __init__.py
│   ├── preprocess_data.py
│   ├── helpers.py
│   └── config.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/debjit-mandal/TrafficVisionAI.git
cd TrafficVisionAI
```

### 2. Create a Virtual Environment and Activate It

```sh
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install the Required Packages

```sh
pip install -r requirements.txt
```

### 4. Download and Extract the GTSRB Dataset

Download the GTSRB dataset and extract it into the `data/GTSRB` directory.

### 5. Train the Traffic Sign Recognition Model

Train the model using the GTSRB dataset:

```sh
python -m src.train_model
```

### 6. Run the Real-Time Driver Assistance System

Start the system to recognize traffic signs, detect lanes, and perform collision detection:

```sh
python -m src.main
```

## Usage

The system will start the webcam feed, process each frame to recognize traffic signs, detect lanes, and perform collision detection. Press `q` to quit.

## Project Components

1. **Traffic Sign Recognition**
   The traffic sign recognition component uses a convolutional neural network (CNN) trained on the GTSRB dataset. The trained model is used to recognize traffic signs in real-time from a webcam feed.

2. **Lane Detection**
   The lane detection component uses OpenCV to detect lane lines in real-time from a webcam feed. It applies Canny edge detection and Hough line transform to identify lane lines.

3. **Collision Detection**
   The collision detection component uses a pre-trained object detection model from TensorFlow Hub to detect objects in real-time from a webcam feed. It draws bounding boxes around detected objects and calculates distances.

## Dependencies

- TensorFlow
- TensorFlow Hub
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## Files and Scripts

1. **src/train_model.py**
   Script to train the traffic sign recognition model using the GTSRB dataset.

2. **src/traffic_sign_recognition.py**
   Script to recognize traffic signs in real-time using the trained model.

3. **src/lane_detection.py**
   Script to detect lane lines in real-time using OpenCV.

4. **src/collision_detection.py**
   Script to detect objects in real-time using a pre-trained model from TensorFlow Hub.

5. **src/main.py**
   Main script to integrate all functionalities: traffic sign recognition, lane detection, and collision detection.

6. **utils/preprocess_data.py**
   Utility script for preprocessing images and loading the GTSRB dataset.

7. **utils/helpers.py**
   Utility script containing helper functions for lane detection.

8. **utils/config.py**
   Configuration file containing paths and settings.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

-------------------------------------------------------------------