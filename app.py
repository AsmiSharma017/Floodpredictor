from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('flood_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/precautions')
def precaution():
    return render_template('precautions.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            features = [float(request.form[x]) for x in ['total_rainfall', 'max_daily_rainfall', 'mean_daily_rainfall', 'duration']]
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
