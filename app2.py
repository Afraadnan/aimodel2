from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from model2 import clf

# Save the model to a file
pickle.dump(clf, open('model2.pkl', 'wb'))

# Load the model from the file
model2 = pickle.load(open('model2.pkl', 'rb'))

print(model2.predict([[20, 40]]))  # Example prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    int_features = [int(data[feature]) for feature in data]
    final_features = [np.array(int_features)]  # Convert to array

    # Make prediction
    prediction = model2.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Time = {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
