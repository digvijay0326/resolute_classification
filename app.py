from flask import Flask, render_template, request
import pandas as pd
import pickle
from src.pipeline.prediction_pipeline import PredictionPipeline
app = Flask(__name__)

# Load the trained model


class CustomData:
    def __init__(self, 
                 T1: int, T2: int, T3: int, T4: int, T5: int, T6: int, T7: int, T8: int, T9: int, T10: int,
                 T11: int, T12: int, T13: int, T14: int, T15: int, T16: int, T17: int, T18: int):
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.T5 = T5
        self.T6 = T6
        self.T7 = T7
        self.T8 = T8
        self.T9 = T9
        self.T10 = T10
        self.T11 = T11
        self.T12 = T12
        self.T13 = T13
        self.T14 = T14
        self.T15 = T15
        self.T16 = T16  
        self.T17 = T17
        self.T18 = T18

    def get_dataframe(self):
        data = {
            'T1': [self.T1],
            'T2': [self.T2],
            'T3': [self.T3],
            'T4': [self.T4],
            'T5': [self.T5],
            'T6': [self.T6],
            'T7': [self.T7],
            'T8': [self.T8],
            'T9': [self.T9],
            'T10': [self.T10],
            'T11': [self.T11],
            'T12': [self.T12],
            'T13': [self.T13],
            'T14': [self.T14],
            'T15': [self.T15],
            'T16': [self.T16],
            'T17': [self.T17],
            'T18': [self.T18]
        }
        return pd.DataFrame(data)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        data = request.form
        custom_data = CustomData(
            T1=int(data["T1"]), T2=int(data["T2"]), T3=int(data["T3"]), T4=int(data["T4"]),
            T5=int(data["T5"]), T6=int(data["T6"]), T7=int(data["T7"]), T8=int(data["T8"]),
            T9=int(data["T9"]), T10=int(data["T10"]), T11=int(data["T11"]), T12=int(data["T12"]),
            T13=int(data["T13"]), T14=int(data["T14"]), T15=int(data["T15"]), T16=int(data["T16"]),
            T17=int(data["T17"]), T18=int(data["T18"])
        )
        df = custom_data.get_dataframe()
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(df)
    
    return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
