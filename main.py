import os
from flask import Flask, render_template
from flask.globals import request
from model import model


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def predict_fn():
    if request.method == "POST":
        result = request.files
        inp = result['file']
        inp.save('input.wav')
        path = os.getcwd()
        aud_path = path+'/input.wav'
        print("hello")
        emotion = model.predict_emotion(
            aud_path, sampling_rate=16384, fft_length=16384, path=path)
        print(emotion)
        return render_template("predict.html", result=emotion[0])
    else:
        return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
