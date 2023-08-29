import requests
import argparse
import os
import json

url = "https://us-central1-cleancurrentscoalition.cloudfunctions.net/upload-images"


def upload_data(device_id, image_file_path, bounding_box_file_path, data):
    files = {
        "device_id": (None, str(device_id)),
        "contents_data": (None, data),
        "image_file": open(image_file_path, "rb"),
        "bounding_box_file": open(bounding_box_file_path, "r"),
    }
    
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Images uploaded successfully.")
    else:
        print(f"Error uploading images. Status code: {response.status_code}")
        print(response.text)

parser = argparse.ArgumentParser(description='Upload trash wheel image, bounding box data, and trash collected data to the server')
parser.add_argument('--device_id', type=int, help='unique id of device')
parser.add_argument('--image_file_path', type=str, help='directory path to image of trash on the trashwheel')
parser.add_argument('--bounding_box_file_path', type=str, help='directory path to bounding box data of the image of trash on the trashwheel')
parser.add_argument('--data', type=str, default='{}', help='collected trash data')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.image_file_path) or not os.path.exists(args.bounding_box_file_path):
        print("Image and bounding box data need to both be present")
    else:
        json_data = json.dumps(json.loads(args.data))
        upload_data(device_id=args.device_id, image_file_path=args.image_file_path, bounding_box_file_path=args.bounding_box_file_path, data=json_data)