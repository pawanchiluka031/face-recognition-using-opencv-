# Face Recognitio Project

This project implements a face recognition system using OpenCV and the Local Binary Patterns Histograms (LBPH) algorithm. It allows you to detect and recognize faces from images.
## Features
- Face detection using Haar Cascades.
- Face recognition with a trained LBPH model.
- Ability to save and load the trained model for future use.

## Project Structure
- Haar Cascade Classifier for face detection.
- LBPH for face recognition.
- Scripts for training and recognizing faces.

## Requirements
- Mine current Python 3.11.2 version
- OpenCV
- NumPy

## Files info
- `faces_train.py`: This script is responsible for training the model using images from the images_for_train folder (which contains the images to be used for training).

- `face_recognization.py`: This script is used to input a photo and test the model. It uses the `face_recognition_model.yml` file, which is the trained model created by the
  Local Binary Patterns Histograms (LBPH) algorithm for face recognition. Along with the model, it also uses `features.npy`, which stores important facial features from the
  training images, and `labels.npy`, which stores the corresponding names of the faces.

- Haar Cascade (`haar_face.xml`): This file, provided by OpenCV on their GitHub, is used to detect faces in images by identifying face-like patterns. It is utilized in both the
  `faces_train.py` and `face_recognization.py` scripts for face detection
