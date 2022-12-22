from infer import InferenceHelper
from ravens_torch.utils import utils
from ravens_torch.tasks import cameras
from PIL import Image
import os
import pickle
import numpy as np

with open('block-insertion/block-insertion-train/action/000000-0.pkl', 'rb') as f1:
    action = pickle.load(f1)
with open('block-insertion/block-insertion-train/color/000000-0.pkl', 'rb') as f1:
    color = pickle.load(f1)
with open('block-insertion/block-insertion-train/depth/000000-0.pkl', 'rb') as f1:
    depth = pickle.load(f1)


obs_0 = {'color': color[0],'depth': depth[0]} 
obs_1 = {'color': color[1],'depth': depth[1]} 

cmap_0, hmap_0 = utils.get_fused_heightmap(obs_0, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
cmap_1, hmap_1 = utils.get_fused_heightmap(obs_1, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
 

print(np.shape(cmap_0))
print(np.shape(hmap_0))
print(np.shape(cmap_1))
print(np.shape(hmap_1))


# img = Image.fromarray(cmap_0, 'RGB')
# img.save('000000-0_pkl_IMAGE/episode.png')
# img.show()
# img = Image.fromarray(cmap_1, 'RGB')
# img.save('000000-0_pkl_IMAGE/goal.png')
# img.show()



# img = Image.fromarray(hmap_0, 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(hmap_0, 'RGB')
# #img.save('my.png')
# img.show()
# print(color)
# print(type(color))
# print(np.shape(color))
# print(action)
# print(type(action))
# print(np.shape(action))


# print(np.shape(color[0]))
# print(np.shape(color[1]))

# print(type(depth))
print(np.shape(depth))
# color_1_0 = color[0][0,:,:,:]
# color_1_1 = color[0][1,:,:,:]
# color_1_2 = color[0][2,:,:,:]
# color_2_0 = color[1][0,:,:,:]
# color_2_1 = color[1][1,:,:,:]
# color_2_2 = color[1][2,:,:,:]

# depth_0 = depth[0][0,:,:]
# depth_1 = depth[1]
# img = Image.fromarray(depth_0 , 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(depth_1 , 'RGB')
# #img.save('my.png')
# img.show()
# # w, h = 512, 512
# # data = np.zeros((h, w, 3), dtype=np.uint8)
# # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# img = Image.fromarray(color_1_0, 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(color_1_1, 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(color_1_2, 'RGB')
# #img.save('my.png')
# img.show()

# img = Image.fromarray(color_2_0, 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(color_2_1, 'RGB')
# #img.save('my.png')
# img.show()
# img = Image.fromarray(color_2_2, 'RGB')
# #img.save('my.png')
# img.show()




model = InferenceHelper(dataset='nyu', device='cpu')

img = Image.open('000000-0_pkl_IMAGE/episode.png') 
#img.show()
img = img.resize((640,480))
img.show()
bin_edges, predicted_depth = model.predict_pil(img)
print(predicted_depth)
print(np.shape(predicted_depth))
print(np.shape(bin_edges))
img = Image.fromarray(predicted_depth[0,0])
img.show()