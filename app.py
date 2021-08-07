import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Performing deserialization using pickle
model = pickle.load(open("admission_model.pkl", 'rb'))


@app.route('/')
def index():
    return render_template(
        'index.html',
        data=[{'UR': 'University Rating'}, {'UR': 1}, {'UR': 2}, {'UR': 3}, {'UR': 4}, {'UR': 5}],
        data1=[{'ReS': 'Research'}, {'ReS': 0}, {'ReS': 1}])


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    input_data = list(request.form.values())
    if int(input_data[0]) & int(input_data[1]) & input_data[3].isdigit() & input_data[4].isdigit() & input_data[5].isdigit() == True:
        pass
    else:
        print(ValueError)

    input_values = [x for x in input_data]
    arr_val = [np.array(input_values)]
    prediction = model.predict(arr_val)
    output = round(prediction[0], 2)*100
    return render_template('index.html', prediction_text=" The Chance of Getting into the University is {} %".format(output),
                           data=[{'UR': 'University Rating'}, {'UR': 1}, {'UR': 2}, {'UR': 3}, {'UR': 4}, {'UR': 5}],
                           data1=[{'ReS': 'Gender'}, {'ReS': 0}, {'ReS': 1}])


if __name__ == '__main__':
    app.run(debug=True)
