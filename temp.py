import torch
import cv2
import os

root = './logs/2023-10-30T20-52-24_example_training-sd_xl_base_ti/checkpoints/'
ckpt_list = os.listdir(root)
ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith('.pt')]
ckpt_list.remove('embeddings.pt')
ckpt_list.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
for ckpt_name in ckpt_list:
    ckpt_path = os.path.join(root, ckpt_name)
    state = torch.load(ckpt_path)
    first_param = state['string_to_param']['*']
    second_param = state['string_to_param_oc']['*']
    print(f'[{ckpt_name}]first_param: {first_param.norm()}, second_param: {second_param.norm()}')
print('done')
# root = './data/cup_ti_v2'
# dst_root = './data/cup_ti_v2_256'
# os.makedirs(dst_root, exist_ok=True)
# image_name_list = os.listdir(root)
# for image_name in image_name_list:
#     image_path = os.path.join(root, image_name)
#     image = cv2.imread(image_path)
#     # resize short edge to 256 and crop
#     h, w, _ = image.shape
#     if h < w:
#         new_h = 256
#         new_w = int(w * 256 / h)
#     else:
#         new_w = 256
#         new_h = int(h * 256 / w)
#     image = cv2.resize(image, (new_w, new_h))
#     h, w, _ = image.shape
#     image = image[h//2-128:h//2+128, w//2-128:w//2+128, :]
#     # save
#     dst_path = os.path.join(dst_root, image_name)
#     cv2.imwrite(dst_path, image)

# a = torch.rand(10, requires_grad=True)
# b = torch.rand(10, requires_grad=False)
# c = torch.rand(20, requires_grad=False)
# output = (b * a).sum()
# output = (c * output).sum()
#
# torch.autograd.grad(output, (a))
