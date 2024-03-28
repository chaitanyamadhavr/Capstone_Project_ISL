# An Efficient Real-Time Word-Level Recognition of Indian Sign Language

This project aims to efficiently recognize Indian Sign Language with higher accuracy and reduced inference time using machine learning techniques.

## Abstract

Sign language facilitates communication for speech and hearing-impaired individuals. This project targets real-time recognition of word-level Indian Sign Language using machine learning. The dataset includes 3,600 videos representing 60 dynamic signs of ISL. Multiple machine learning models were assessed, with the Convolutional LSTM model achieving 99.88% accuracy. The technology aims to enrich the lives of the hearing-impaired by enhancing communication through sign language recognition.

## Dataset

The curated dataset consists of 3,600 videos depicting 60 distinct signs of ISL, each with 60 videos. These signs involve both hand gestures captured at 640 x 720 pixels resolution. Reference for the signs were drawn from the "ISLRTC NEW DELHI" an official government-backed YouTube channel dedicated to Indian Sign Language.

## System Architecture

### Feature Extractor

This research utilizes Mediapipe Holistic, a tool by Google, to extract 75 landmarks representing both hand and pose elements crucial for interpreting Indian Sign Language (ISL) gestures accurately. These landmarks, represented by normalized coordinates, are organized into numpy arrays for efficient data handling. By capturing 50 frames per video and structuring data per sign, this approach facilitates precise analysis and model development. Integrating Mediapipe Holistic ensures standardized representation and empowers models to discern subtle variations in ISL gestures.

### Models and Results

Several machine learning models were evaluated, with Convolutional LSTM model achieving the highest accuracy of 99.88%.

### Real-time Output


## Conclusion

In this research, we created a dataset of 3600 videos representing 60 dynamic signs in Indian Sign Language (ISL). Using Mediapipe as a feature extractor, we evaluated nine models for ISL recognition, achieving reduced inference time, real-time capability, adaptability to variable lighting conditions, and robustness against occlusions, with lower processing power demand. These models demonstrate efficiency even with smaller datasets, resulting in quicker training times averaging up to 12 minutes for all 3600 videos. Additionally, they are independent of signer-specific parameters like skin tone, gender, and movements, excelling at recognizing complex signs and handling multiple signers. Real-time recognition, integrated with a tkinter interface, showcases practical applications. These achievements lay a foundational framework for robust sign language recognition technologies, poised to significantly empower individuals with hearing impairments through accurate and efficient recognition of sign languages.

Authors:
<br>
Chaitanya Madhav R  PES1UG20CS634
<br>
Prasad S Shet       PES1UG20CS575
<br>
Mohan Murali K S    PES1UG20CS563
<br>
Likhith R           PES1UG20CS659
<br>
