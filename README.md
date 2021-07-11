# SER-project
S8 project on emotion recognition from speech


#-initialisation

pipenv shell
pipenv install


#-To test 

python test.py


description : test.py contains an variable named aud which is initalised with a audio path. Running the python file prints the expected emotion and predicted emotion


#-Site

python main.py

description : main.py runs a site build with flask. The site contains a input field which accepts audio files. It sends the file back to server. The server then returns an emotion as a response.


#- Accuracy

cd prediction
python accuracy.py

description : accuracy.py returns the accuracy of the trained model

