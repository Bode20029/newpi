import requests

class LineNotifier:
    def __init__(self, token):
        self.token = token
        self.api_url = 'https://notify-api.line.me/api/notify'

    def initialize(self):
        # You can add any initialization logic here if needed
        return True

    def send_notification(self, message):
        headers = {'Authorization': f'Bearer {self.token}'}
        payload = {'message': message}
        response = requests.post(self.api_url, headers=headers, data=payload)
        return response.status_code == 200

    def send_image(self, message, image_path):
        headers = {'Authorization': f'Bearer {self.token}'}
        payload = {'message': message}
        files = {'imageFile': open(image_path, 'rb')}
        response = requests.post(self.api_url, headers=headers, data=payload, files=files)
        return response.status_code == 200