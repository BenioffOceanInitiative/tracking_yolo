import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='exp5_200.pt')
model.conf = 0.1  # Confidence threshold (0-1)

img = "hc1.jpg"

results = model(img)


results.show()
results.save()
# print names and counts
print(results.pandas().xyxy[0].name.value_counts())
# print(results.pandas().xyxy[0])  # print img1 predictions (pixels)
