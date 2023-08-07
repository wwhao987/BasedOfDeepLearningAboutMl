import requests

url = 'http://121.40.96.93:9999/predict?features=2.25,4.35,0.45,0.55;1.25,1.35,1.1,1.2'
url = 'http://127.0.0.1:9999/predict?features=1,2,3,4;0.1,0.2,0.5,0.3;2.1,3.5,1.2,1.3'
response = requests.get(url)
if response.status_code == 200:
    result = response.json()
    if result['code'] != 0:
        print("调用服务异常!")
    print(result)
    print(type(result))
print(response)
