import requests
import json

url = 'http://127.0.0.1:8501/v1/models/chatbot:predict'
payload = {"instances": [["Muchas gracias!"]]}
data = json.dumps(payload)
headers = {
  'Content-Type': 'application/json'
}

json_response = requests.post(url, headers=headers, data=data)
result = json.loads(json_response.text)['predictions'][0]
print(result)


'''
La predicción del bot está compuesta de los siguientes campos:

predictions: {
    label: ..., # tipo de respuesta (tag)
    score: ..., # Valor 0 a 1 que indica la precisión de esa respuesta
    messsage: ... # Mensaje del bot como respuesta (texto)
}
'''
