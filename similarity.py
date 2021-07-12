
import pandas as pd
from semantic_text_similarity.models import WebBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction
data_df = pd.read_csv("data/test.csv",header=0,encoding="utf-8")

simi_input_0 = []
simi_input_1 = []
labels = []
lines = data_df.values.tolist()
for line in lines:
    simi_input_0.append((line[1],line[2]))
    simi_input_1.append((line[3], line[4]))
    labels.append(line[-1])

simi_res_0 = web_model.predict(simi_input_0)
simi_res_1 = web_model.predict(simi_input_1)

count = 0
preds = []

s0_list = []
s1_list = []
for label,s0,s1 in zip(labels,simi_res_0,simi_res_1):
    s0_list.append(s0)
    s1_list.append(s1)
    if s0>s1:
        preds.append(0)
        if int(label)==0:
            count+=1
    elif s0<s1:
        preds.append(1)
        if int(label)==1:
            count+=1

print("A semantic_text_similarity model based on BERT without using any COPA data achieves the accuracy of:  ",count/500.00)


