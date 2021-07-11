import pickle

with open("/home/sidharth/Desktop/S8project/code/site/Emotion_Voice_Detection_Model_dataset.pkl", 'rb') as file:
    model = pickle.load(file)

with open("/home/sidharth/Desktop/S8project/SER project/final_x_train.pkl", 'rb') as file:
    x_train = pickle.load(file)

with open("/home/sidharth/Desktop/S8project/SER project/final_x_test.pkl", 'rb') as file:
    x_test = pickle.load(file)

with open("/home/sidharth/Desktop/S8project/SER project/final_y_train.pkl", 'rb') as file:
    y_train = pickle.load(file)

with open("/home/sidharth/Desktop/S8project/SER project/final_y_test.pkl", 'rb') as file:
    y_test = pickle.load(file)

model.fit(x_train, y_train)
pred = model.predict(x_test)

length = len(pred)
i = 0
count = 0
while(i < length):
    if(pred[i] == y_test[i]):
        count = count+1
    i = i+1
accuracy = (count/length)*100
print(accuracy)
