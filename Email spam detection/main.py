from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
    
# Load the model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/', methods=["GET", "POST"])
def main_function():
    if request.method == "POST":
        text = request.form
        emails = text['email']
        
        # Convert text to feature vectors
        list_email = [emails]
        email_features = vectorizer.transform(list_email)
        
        # Predict
        output = model.predict(email_features)[0]
        
        prediction = 'Ham' if output == 1 else 'Spam'
        
        return render_template("show.html", prediction=prediction)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
