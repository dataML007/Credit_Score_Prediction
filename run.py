import sys
import requests
import json
import pandas as pd

url = 'http://0.0.0.0:5000/api/'
if len(sys.argv) < 1:
    print("Please provide a filename to continue")
    exit(0)

score_output_file = 'score_output.csv'
filename = sys.argv[1]
df = pd.read_csv(filename)
data = df.to_dict('records')
j_data = json.dumps(data)
headers = {'Content-Type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
output = json.loads(r.text)
pred = output['prediction']
rmse_val = output['rmse_val']
acc_val = output['acc_val']
r2_val = output['r2_val']
print("\n")
print("RMSE - ", rmse_val)
print("\n")
print("Accuracy - ", acc_val)
print("\n")
print("R squared - ", r2_val)
pred_df = pd.DataFrame(pred, columns=['prediction'])
pred_df.to_csv(score_output_file, index=False, header=False)
print("Scoring Complete")