import requests 
import json 


# input a text 
input_text = str(input("Enter a text to classify: "))
# headers to be sent
headers = {
            "accept":"application/json",
            "Content-Type":"application/json"
            }
# the payload
data = {
        "text":input_text
        }

# URL
url = 'http://127.0.0.1:8000/predict'

# the response 
res = requests.post(url, headers=headers, data=json.dumps(data))

# get the content 
res = res.content 

# make it into a json file 
res_json = json.loads(res.decode('utf-8'))

# print the items 
for k, v in res_json.items():

    if type(v) == dict:
        print(k)
        for f,g in v.items():
            print(f, g)
    else:
        print(k, v,'\n')
