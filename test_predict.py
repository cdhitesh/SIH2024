import requests

URL = 'http://localhost:8000/predict'
img_path = 'Indian/A/1.jpg'  # update to an available image path

with open(img_path, 'rb') as f:
    files = {'file': ('test.jpg', f, 'image/jpeg')}
    r = requests.post(URL, files=files)
    print(r.status_code, r.text)
