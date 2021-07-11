import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_dict = {
    "neutral": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "calm": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "happy": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "sad": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "angry": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "fearful": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "disgust": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    },
    "surprised": {
        "neutral": 0,
        "calm": 0,
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgust": 0,
        "surprised": 0
    }
}


itr = 0

pklfile = os.path.abspath(os.path.join(os.pardir)) + \
    '/Emotion_Voice_Detection_Model_dataset.pkl'
with open(pklfile, 'rb') as file:
    Emotion_Voice_Detection_Model = pickle.load(file)

pklfile = os.path.abspath(os.path.join(os.pardir)) + \
    "/final_x_train.pkl"
with open(pklfile, 'rb') as file:
    x_values = pickle.load(file)

pklfile = os.path.abspath(os.path.join(os.pardir)) + \
    "/final_x_test.pkl"
with open(pklfile, 'rb') as file:
    x_test = pickle.load(file)

pklfile = os.path.abspath(os.path.join(os.pardir)) + \
    "/final_y_test.pkl"
with open(pklfile, 'rb') as file:
    y = pickle.load(file)

scaler = MinMaxScaler()
x_new = scaler.fit(x_values)

for i in range(len(x_test)):

    emotion = y[i]

    feature = [x_test[i]]
    feature = np.array(feature)
    feature = scaler.transform(feature)

    pred_emotion = Emotion_Voice_Detection_Model.predict(feature)[0]

    emotion_dict[emotion][pred_emotion] = emotion_dict[emotion][pred_emotion] + 1

    if itr == 100:
        print(emotion_dict)
    itr = itr + 1


print(emotion_dict)
