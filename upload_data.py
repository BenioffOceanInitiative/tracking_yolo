import requests
import argparse
import os

url = "https://us-central1-cleancurrentscoalition.cloudfunctions.net/upload-images"


def upload_data(device_id, raw_image_path, annotated_image_path, data):
    files = {
        "device_id": (None, device_id),
        "contents_data": (None, data),
        "image_file": ("", open(raw_image_path, "rb")),
        "bounding_box_file": ("", open(annotated_image_path, "rb")),
    }
    
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Images uploaded successfully.")
    else:
        print(f"Error uploading images. Status code: {response.status_code}")
        print(response.text)



parser = argparse.ArgumentParser(description='Upload raw and annotated images and trash collected data to the server')
parser.add_argument('--device_id', type=int, help='unique id of device')
parser.add_argument('--image_path', type=argparse.FileType('rb'), help='directory path to image of trash on the trashwheel')
parser.add_argument('--bounding_box_file_path', type=argparse.FileType('rb'), help='directory path to bounding box data of the image of trash on the trashwheel')
parser.add_argument('--data', type=str, default={}, help='collected trash data')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.image_path) or not os.path.exists(args.bounding_box_file_path):
        print("Image and bounding box data need to both be present")
    else:
        upload_data(device_id=args.device_id, image_path=args.image_path, bounding_box_file_path=args.bounding_box_file_path, data=args.data)