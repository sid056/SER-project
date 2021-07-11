import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from model import model
from prediction import confusion

path = os.getcwd()

with open(path+"/Emotion_Voice_Detection_Model_dataset.pkl", 'rb') as file:
    Emotion_Voice_Detection_Model = pickle.load(file)

with open(path+"/final_x.pkl", 'rb') as file:
    x_values = pickle.load(file)


scaler = MinMaxScaler()
scaler.fit(x_values)

aud = path+"/dataset/03-01-04-02-02-02-22.wav"
basename = os.path.basename(aud)
actual_emotion = confusion.emotions[basename.split("-")[2]]
features = model.mfcc(aud, 16384, 16384, num_cepstral=19)
feature = [features]
feature = np.array(feature)
print(feature)
feature = scaler.transform(feature)
print(feature)

emotion = Emotion_Voice_Detection_Model.predict(feature)
print("actual emotion : ", actual_emotion)
print("predicted_emotion : ", emotion[0])
