from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as m_file:
    model = pickle.load(m_file)

with open('vectorizer.pkl', 'rb') as v_file:
    vectorizer = pickle.load(v_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    user_input = ""
    
    if request.method == 'POST':
        user_input = request.form['message']
        if user_input.strip() != "":
            # Transform and Predict
            data_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(data_vectorized)[0]
            
            # Formulate Output Label
            prediction_text = "🚨 SPAM ALERT!" if prediction == 1 else "✅ HAM (Legitimate Message)"

    return render_template('index.html', prediction=prediction_text, text_entered=user_input)

if __name__ == '__main__':
    app.run(debug=True)