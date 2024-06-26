from flask import Flask, request, jsonify, render_template
import pickle
## Load the model
input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

# @app.route('/', methods=['GET']) # Homepage
# def home():

#     return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # customer = request.get_json()
        customer = request.form.to_dict()

        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5  # Assuming a threshold of 0.5 for churn prediction
        result = {
            'churn_probability': round(float(y_pred),2),
            'churn': bool(churn)
        }

        # Render the same template with the prediction results
        return render_template('index.html', prediction_results = result)
    # For GET request, render the template without results    
    return render_template('index.html')
   


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True)