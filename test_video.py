import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from data_utils import divide_image, join_images
from model import Generator
import gc

# gc.collect()
parser = argparse.ArgumentParser(description="Test Single Image")
parser.add_argument('--upscale_factor', default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--video_name', type=str, help='test low resolution video name')
parser.add_argument('--image_mode', type=bool, default=True, help='test low resolution video name')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
VIDEO_NAME = opt.video_name
MODEL_NAME = 'netG_epoch_4_100.pth'
image_mode = opt.image_mode
torch.cuda.empty_cache()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
torch.cuda.memory_summary(device="cuda", abbreviated=False)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


video = cv2.VideoCapture(VIDEO_NAME)
# ret, frame = video.read()
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width*UPSCALE_FACTOR, frame_height*UPSCALE_FACTOR)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = video.get(cv2.CAP_PROP_FPS)
outv = cv2.VideoWriter('output/filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         fps, size)
count = 0
while True:
  ret, frame = video.read()
  if not ret:
            break
  count += 1
  images = divide_image(frame)
  out_images = []
  for image in images:
      image = Image.fromarray(image).convert("RGB")
      image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
      if TEST_MODE:
          image = image.cuda()
      out = model(image)
      out_image = ToPILImage()(out[0].data.cpu())
      out_images.append(np.array(out_image))
  img = join_images(out_images)
  print(count)
  if image_mode:
     cv2.imwrite(f"output/{count}.png", img)
  else:
     outv.write(img)

print("*****inferencing finished*****")
video.release()
outv.release()