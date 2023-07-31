import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk
import re
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.metrics import f1_score,accuracy_score,classification_report
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from flask import Flask, request, jsonify

# Reading the parquet dataset
print("Starting to read")
df = pd.read_parquet('D:\kavida.ai\kavida.ai\query_result_2000.993493Z (1).parquet', engine='fastparquet')
print("Reading done")
# Dropping the duplicates from the dataset
df = df.drop_duplicates()
# The below code deals with the list value present in the news_list column and it it converting it into only string dtype.
l = []
for i in df['news_list']:
    a = i[2:-2]
    l.append(a)
# Overwriting the values of feature news_list by newly generated data stored in l.
df['news_list'] = l
df3 = df.copy()
nltk.download('stopwords')
stop_list = set(stopwords.words("english"))

### combining the title_new and paragraph_new feature into a single column
df3['combined_text'] = df3['title'] + df3['paragraph']


label_encoder =  preprocessing.LabelEncoder()
df3["news_l_e"] = df3['news_list']
df3['news_l_e'] = label_encoder.fit_transform(df3['news_l_e'])

# Mapping all  the news_l_e feature values to a categorical variable
df3['news_l_e_map'] = df3.news_l_e.map({0:"A",1:"B",2:"C",3:"D",4:"E",5:"F"})
dic_num_news = {0:"Commodities",1:"Compliance",2:"Delays",3:"Environmental",4:"Financial Health",5:"Supplier Market"}
dic = {"A":"Commodities","B":"Compliance","C":"Delays","D":"Environmental","E":"Financial Health","F":"Supplier Market"}
print("Setting the device to GPU/CPU")

# Set the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Setting the device to GPU/CPU -> DONE")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_model(model_class,num_labels, path):
    model = model_class.from_pretrained('bert-base-uncased',num_labels=num_labels)
    model.load_state_dict(torch.load(path),strict=False)
    model.to(device)  # Move the model back to the appropriate device
    return model


# Load the pre-trained model
loaded_model = load_model(BertForSequenceClassification, len(label_encoder.classes_), 'D:\kavida.ai\kavida.ai\pretrained_model.pt')
# Create the Flask app
app = Flask(__name__)



def predict_single_input(model, input_paragraph):
        model.eval()
        inputs = tokenizer(input_paragraph, truncation=True, padding=True, max_length=256, return_tensors='pt')
        input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            predicted_label = label_encoder.classes_[predicted.item()]

        return predicted_label  

# Define the route for prediction
@app.route("/predict", methods=["POST"])
def predict():  
    if request.method == "POST":
        data = request.json
        text = data["text"]
        Predicted_Label = predict_single_input(loaded_model,text)

        return jsonify({"Predicted Label": Predicted_Label})

# Run the app if this file is executed
if __name__ == "__main__":
    app.run(debug=True,port = 5000)
