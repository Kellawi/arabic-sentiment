from flask import Flask, render_template
from keras.models import model_from_json
import pandas as pd
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences
from flask import request


df = pd.read_csv('reviews.csv')
app = Flask(__name__)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df.reviews.astype(str))

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    sentence = str(request.form["sentence"])
    i = []
    i.append(sentence)

    i = tokenizer.texts_to_sequences(i)
    i = pad_sequences(i, maxlen = 452)
    prediction = loaded_model.predict(i)  # this returns a list e.g. [127.20488798], so pick first element [0]
    
    if prediction[0][0] > prediction[0][1]:
        output = 'سلبية'
    else:
        output = 'إيجابية'

    return render_template('index.html', prediction_text=f'الجملة التالية:\n\"{sentence}\"\n هي جملة: \n{output}')

if __name__ == "__main__":
    app.run(debug=True)