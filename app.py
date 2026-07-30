from flask import Flask, render_template, request
from prediction import get_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment_result = ""
    user_input = ""
    pos_prob = 0
    neg_prob = 0
    
    if request.method == 'POST':
        user_input = request.form['tweet_text']
        
        # Unpack the three variables returned by our model
        sentiment_result, pos_prob, neg_prob = get_sentiment(user_input)
        
    return render_template('index.html', 
                           text=user_input, 
                           sentiment=sentiment_result,
                           pos_prob=pos_prob,
                           neg_prob=neg_prob)

if __name__ == '__main__':
    app.run(debug=True)