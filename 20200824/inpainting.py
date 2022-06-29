# https://github.com/akmtn/pytorch-siggraph2017-inpainting

# must be run with pytorch version 0.4.1
# download the wheels from here: https://download.pytorch.org/whl/cpu/torch_stable.html
# pip install torch-0.4.1-cp37-cp37m-win_amd64.whl # --force-reinstall
# pip install "torchvision-0.4.1+cpu-cp37-cp37m-win_amd64.whl" # --force-reinstall
import torch
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import matplotlib.pylab as plt


def tensor2image(src):
    out = src.copy() * 255
    out = out.transpose((1, 2, 0)).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def image2tensor(src):
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2,0,1)).astype(np.float64) / 255
    return out


image_path = 'zebra.png'
mask_path = 'inpaint_mask.png'
image_path = 'lena.png'
mask_path = 'lena_mask.png'
# image_path = 'inter.jpg'
# mask_path = 'inter_mask.jpg'
# image_path = 'wall.jpg'
# mask_path = 'wall_mask.jpg'

# download model from http://hi.cs.waseda.ac.jp/~iizuka/data/completionnet_places2.t7
model_path = 'completionnet_places2.t7'
gpu = torch.cuda.is_available()

# 학습된 모델 로드
# load_lua는 더 이상 지원 안함
data = load_lua(model_path,long_size=8)
model = data.model
model.evaluate()

# 로드된 이미지 크기를 4로 나누어지게 크기 조절
image = cv2.imread(image_path)
image = cv2.resize(image, (4*(image.shape[1]//4), 4*(image.shape[0]//4)))
# torch에서 사용하는 tensor로 변경
I = torch.from_numpy(image2tensor(image)).float()

# 마스크 크기를 4로 나누어지게 크기 조절 및 이진화
mask = cv2.imread(mask_path)
mask = cv2.resize(mask, (4*(mask.shape[1]//4), 4*(mask.shape[0]//4)))
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255
mask[mask <= 0.5] = 0.0
mask[mask > 0.5] = 1.0

# torch에서 사용하는 tensor로 변경 및 한 차원을 늘린다.
M = torch.from_numpy(mask).float()
M = M.view(1, M.size(0), M.size(1))
assert I.size(1) == M.size(1) and I.size(2) == M.size(2)

# 평균을 빼서 이미지를 가운데 위치 시킨다.
for i in range(3):
    I[i, :, :] = I[i, :, :] - data.mean[i]

# 마스크를 3채널로 늘린다.
M3 = torch.cat((M, M, M), 0)
# 마스크가 적용된 부분은 하얀색이 되도록 마스크를 적용한다.
im = I * (M3*(-1)+1)

# 마스크가 적용된 이미지와 마스크를 input으로 만든 뒤 input의 차원에 맞게 조절한다.
input = torch.cat((im, M), 0)
input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

if gpu:
    print('using GPU...')
    model.cuda()
    input = input.cuda()

# 학습된 데이터를 바탕으로 이미지 완성
res = model.forward(input)[0].cpu()

# 이미지를 원래 위치로 조절
for i in range(3):
    I[i, :, :] = I[i, :, :] + data.mean[i]

# 완성된 부분 res를 마스크에 씌워 마스크 처리된 원본 이미지에 합성
out = res.float()*M3.float() + I.float()*(M3*(-1)+1).float()

image[mask > 0.5] = 255

plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Incomplete Image', size=20)
plt.subplot(122), plt.imshow(cv2.cvtColor(tensor2image(out.numpy()), cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Completed Image (with CompletionNet)', size=20)

plt.show()

print('Done')