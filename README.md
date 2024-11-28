# ASL (American Sign Language) Recognition System

A real-time hand gesture recognition system that detects and classifies American Sign Language (ASL) alphabets using computer vision and machine learning.

## Features
- Real-time hand gesture detection
- Recognition of ASL alphabets (A-Y)
- Live video feed with visual feedback
- Bounding box around detected hand
- Display of predicted alphabet

## Prerequisites
- Python 3.10 or later
- Webcam
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

git clone https://github.com/RuturajKhondre/ASL-Recognition-Project.git

cd ASL-Recognition-Project

2. Create and activate a virtual environment (recommended):

# Windows

python -m venv venv

venv\Scripts\activate


3. Install required packages:

pip install -r requirements.txt


## Usage

### 1. Data Collection
Run the data collection script to capture hand gesture images:

python collect_imgs.py
- Follow the on-screen instructions
- Press 'Q' to start capturing images for each alphabet
- The script will capture multiple images for each gesture
- Images are saved in the `data` directory

### 2. Create Dataset
Process the collected images to create the training dataset:


python create_dataset.py

- This will process all images in the `data` directory
- Creates `data.pickle` file containing processed hand landmarks

### 3. Train Model
Train the machine learning model on the processed dataset:


python train_classifier.py
- Splits data into training and testing sets
- Trains a Random Forest Classifier
- Saves the trained model as `model.p`
- Displays training accuracy

### 4. Real-time Recognition
Start the real-time recognition system:


python inference_classifier.py
- Opens webcam feed
- Detects hand gestures in real-time
- Displays predicted alphabet
- Press 'Q' to quit

## Project Structure


│

├── collect_imgs.py        # Image collection script


├── create_dataset.py      # Dataset creation script

├── train_classifier.py    # Model training script

├── inference_classifier.py # Real-time recognition script

├── requirements.txt       # Required Python packages

└── README.md             # Project documentation



## Technologies Used
- OpenCV (cv2) - Computer vision and image processing
- MediaPipe - Hand landmark detection
- scikit-learn - Machine learning (Random Forest Classifier)
- NumPy - Numerical computations
- Python - Programming language





## Author
[RuturajKhondre]

## Contact
- GitHub: [RuturajKhondre](https://github.com/RuturajKhondre)
