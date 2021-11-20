import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'male':0 ,'age':80 ,'education':3 ,'currentSmoker':1 ,'cigsPerDay':50 ,'prevalentHyp':1 ,'totChol':500 ,'sysBP':150 ,'diaBP':105 ,'BMI':28.58 ,'heartRate':0 ,'glucose':200})

print(r.json())