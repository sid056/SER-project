import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from model import model
with open("/home/sidharth/Desktop/S8project/code/site/Emotion_Voice_Detection_Model_dataset.pkl", 'rb') as file:
    Emotion_Voice_Detection_Model = pickle.load(file)

with open("/home/sidharth/Desktop/S8project/SER project/final_x.pkl", 'rb') as file:
    x_values = pickle.load(file)


scaler = MinMaxScaler()
scaler.fit(x_values)

aud = "/home/sidharth/Desktop/S8project/SER project/dataset/03-01-01-01-01-01-14.wav"
features = model.mfcc(aud, 16384, 16384, num_cepstral=19)
feature = [features]
feature = np.array(feature)
print(feature)
feature = scaler.transform(feature)
print(feature)

emotion = Emotion_Voice_Detection_Model.predict(feature)
print(emotion)
