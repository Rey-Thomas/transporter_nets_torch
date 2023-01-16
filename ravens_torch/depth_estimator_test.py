#from infer import InferenceHelper
from ravens_torch.utils import utils
from ravens_torch.tasks import cameras
from PIL import Image
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt


visualisation=True
kitti=True
nyu=True


with open('block-insertion/block-insertion-train/action/000000-0.pkl', 'rb') as f1:
    action = pickle.load(f1)
with open('block-insertion/block-insertion-train/color/000000-0.pkl', 'rb') as f1:
    color = pickle.load(f1)
with open('block-insertion/block-insertion-train/depth/000000-0.pkl', 'rb') as f1:
    depth = pickle.load(f1)




# with open('towers-of-hanoi/towers-of-hanoi-train/action/000000-0.pkl', 'rb') as f1:
#     action = pickle.load(f1)
# with open('towers-of-hanoi/towers-of-hanoi-train/color/000000-0.pkl', 'rb') as f1:
#     color = pickle.load(f1)
# with open('towers-of-hanoi/towers-of-hanoi-train/depth/000000-0.pkl', 'rb') as f1:
#     depth = pickle.load(f1)



# with open('stack-block-pyramid/stack-block-pyramid-train/action/000000-0.pkl', 'rb') as f1:
#     action = pickle.load(f1)
# with open('stack-block-pyramid/stack-block-pyramid-train/color/000000-0.pkl', 'rb') as f1:
#     color = pickle.load(f1)
# with open('stack-block-pyramid/stack-block-pyramid-train/depth/000000-0.pkl', 'rb') as f1:
#     depth = pickle.load(f1)


print(f'type  action {type(action)}, len: {len(action)} action:{action}')
print(f'type  color {type(color)}, len: {np.shape(color)}')
print(f'type  depth {type(depth)}, len: {np.shape(depth)}')

print(np.shape(depth[0][0]))
depth_0 = np.expand_dims(depth[0][0],axis=0)
depth_1 = np.expand_dims(depth[-1][0],axis=0)
color_0 = np.expand_dims(color[0][0],axis=0)
color_1 = np.expand_dims(color[-1][0],axis=0)
print(np.shape(depth_0))
obs_0 = {'color': color[0],'depth': depth[0]} 
obs_1 = {'color': color[-1],'depth': depth[-1]} 
# obs_0 = {'color': color_0,'depth': depth_0} 
# obs_1 = {'color': color_1,'depth': depth_1}


if visualisation:
    img = Image.fromarray(color[0][0].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-begin-episode-cam0.png')
    img.show()
    img = Image.fromarray(color[0][1].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-begin-episode-cam1.png')
    img.show()
    img = Image.fromarray(color[0][2].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-begin-episode-cam2.png')
    img.show()
    img = Image.fromarray(color[-1][0].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-end-episode-cam0.png')
    img.show()
    img = Image.fromarray(color[-1][1].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-end-episode-cam1.png')
    img.show()
    img = Image.fromarray(color[-1][2].squeeze(), 'RGB')
    img.save('depth/block-insertion-0-end-episode-cam2.png')
    img.show()
    plt.imshow(depth[0][0], cmap='plasma')
    plt.savefig('depth/block-insertion-0-begin-episode-cam0_depth.png')
    plt.imshow(depth[0][1], cmap='plasma')
    plt.savefig('depth/block-insertion-0-begin-episode-cam1_depth.png')
    plt.imshow(depth[0][2], cmap='plasma')
    plt.savefig('depth/block-insertion-0-begin-episode-cam2_depth.png')
    plt.imshow(depth[-1][0], cmap='plasma')
    plt.savefig('depth/block-insertion-0-end-episode-cam0_depth.png')
    plt.imshow(depth[-1][1], cmap='plasma')
    plt.savefig('depth/block-insertion-0-end-episode-cam1_depth.png')
    plt.imshow(depth[-1][2], cmap='plasma')
    plt.savefig('depth/block-insertion-0-end-episode-cam2_depth.png')



cmap_0, hmap_0 = utils.get_fused_heightmap(obs_0, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
cmap_1, hmap_1 = utils.get_fused_heightmap(obs_1, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
 
# print(np.shape(cmap_0))
# print(np.shape(hmap_0))
# print(np.shape(cmap_1))
# print(np.shape(hmap_1))

print(f'hmap_0 max: {np.max(hmap_0)}, hmap_0 min: {np.min(hmap_0)}, hmap_0 sum: {np.sum(hmap_0)}, hmap_0 type: {type(hmap_0)}, hmap_0 shape: {np.shape(hmap_0)} \n {hmap_0}')

img = Image.fromarray(cmap_0, 'RGB')
img.save('depth/begin_episode_homographie.png')
img.show()
img = Image.fromarray(cmap_1, 'RGB')
img.save('depth/end_episode_homographie.png')
img.show()

# plt.imshow(cmap_0)
# plt.savefig('depth/begin_episode_homographie.png')
# plt.imshow(cmap_1)
# plt.savefig('depth/end_episode_homographie.png')

plt.imshow(hmap_0, cmap='plasma')
plt.savefig('depth/begin_episode_homographie_depth.png')
plt.imshow(hmap_1, cmap='plasma')
plt.savefig('depth/end_episode_homographie_depth.png')


if kitti:
    from infer import InferenceHelper
    model = InferenceHelper(dataset='kitti', device='cuda')

    img = Image.open('depth/block-insertion-0-begin-episode-cam0.png') 
    #img.show()
    #img = img.resize((640,480))
    #img.show()
    bin_edges, predicted_depth_begin_0 = model.predict_pil(img)
    #print(predicted_depth)
    #print(np.shape(predicted_depth))
    #print(np.shape(bin_edges))
    #img = Image.fromarray(predicted_depth[0,0])
    #img.show()


    plt.imshow(predicted_depth_begin_0[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam0_depth_estimated_kitti.png")

    img = Image.open('depth/block-insertion-0-begin-episode-cam1.png')
    bin_edges, predicted_depth_begin_1 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_1[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam1_depth_estimated_kitti.png")

    img = Image.open('depth/block-insertion-0-begin-episode-cam2.png')
    bin_edges, predicted_depth_begin_2 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_2[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam2_depth_estimated_kitti.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam0.png')
    bin_edges, predicted_depth_end_0 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_0[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam0_depth_estimated_kitti.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam1.png')
    bin_edges, predicted_depth_end_1 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_1[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam1_depth_estimated_kitti.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam2.png')
    bin_edges, predicted_depth_end_2 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_2[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam2_depth_estimated_kitti.png")

    img = Image.open('depth/begin_episode_homographie.png') 
    img.show()
    img = img.resize((640,480))
    img.show()
    bin_edges, predicted_depth_begin_0 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_0[0][0], cmap='plasma')
    plt.savefig("depth/begin_episode_homographie_estimated_depth_kitti.png")

    img = Image.open('depth/end_episode_homographie.png') 
    img.show()
    img = img.resize((640,480))
    img.show()
    bin_edges, predicted_depth_begin_0 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_0[0][0], cmap='plasma')
    plt.savefig("depth/end_episode_homographie_estimated_depth_kitti.png")
    
    print(np.shape(predicted_depth_begin_0))
    depth_estimated_0=np.concatenate((predicted_depth_begin_0[0],predicted_depth_begin_1[0],predicted_depth_begin_2[0]),axis=0)
    print(np.shape(depth_estimated_0))
    depth_estimated_1=np.concatenate((predicted_depth_end_0[0],predicted_depth_end_1[0],predicted_depth_end_2[0]),axis=0)
    print(np.shape(depth_estimated_1))
    obs_0 = {'color': color[0],'depth': depth_estimated_0} 
    obs_1 = {'color': color[-1],'depth': depth_estimated_1} 
    print(np.shape(color[0][0]))
    # obs_0 = {'color': color[0],'depth': predicted_depth_begin_0[0]} 
    # obs_1 = {'color': color[1],'depth': predicted_depth_end_0[0]} 
    cmap_0, hmap_0 = utils.get_fused_heightmap(obs_0, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    cmap_1, hmap_1 = utils.get_fused_heightmap(obs_1, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    print(np.shape(cmap_0))
    print(np.shape(hmap_0))
    print(np.shape(cmap_1))
    print(np.shape(hmap_1))

    print(f'hmap_0 max: {np.max(hmap_0)}, hmap_0 min: {np.min(hmap_0)}, hmap_0 sum: {np.sum(hmap_0)}, hmap_0 type: {type(hmap_0)}, hmap_0 shape: {np.shape(hmap_0)} \n {hmap_0}')

    img = Image.fromarray(cmap_0, 'RGB')
    img.save('depth/begin_episode_homographie.png')
    img.show()
    img = Image.fromarray(cmap_1, 'RGB')
    img.save('depth/end_episode_homographie.png')
    img.show()

    plt.imshow(cmap_0)
    plt.savefig('depth/begin_episode_homographie.png')
    plt.imshow(cmap_1)
    plt.savefig('depth/end_episode_homographie.png')

    plt.imshow(hmap_0, cmap='plasma')
    plt.savefig('depth/begin_episode_homographie_depth_kitti.png')
    plt.imshow(hmap_1, cmap='plasma')
    plt.savefig('depth/end_episode_homographie_depth_kitti.png')

if nyu:
    from infer import InferenceHelper
    model = InferenceHelper(dataset='nyu', device='cuda')

    img = Image.open('depth/block-insertion-0-begin-episode-cam0.png') 
    #img.show()
    #img = img.resize((640,480))
    #img.show()
    bin_edges, predicted_depth_begin_0 = model.predict_pil(img)
    #print(predicted_depth)
    #print(np.shape(predicted_depth))
    #print(np.shape(bin_edges))
    #img = Image.fromarray(predicted_depth[0,0])
    #img.show()


    plt.imshow(predicted_depth_begin_0[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam0_depth_estimated_nyu.png")

    img = Image.open('depth/block-insertion-0-begin-episode-cam1.png')
    bin_edges, predicted_depth_begin_1 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_1[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam1_depth_estimated_nyu.png")

    img = Image.open('depth/block-insertion-0-begin-episode-cam2.png')
    bin_edges, predicted_depth_begin_2 = model.predict_pil(img)
    plt.imshow(predicted_depth_begin_2[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-begin-episode-cam2_depth_estimated_nyu.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam0.png')
    bin_edges, predicted_depth_end_0 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_0[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam0_depth_estimated_nyu.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam1.png')
    bin_edges, predicted_depth_end_1 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_1[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam1_depth_estimated_nyu.png")

    img = Image.open('depth/block-insertion-0-end-episode-cam2.png')
    bin_edges, predicted_depth_end_2 = model.predict_pil(img)
    plt.imshow(predicted_depth_end_2[0][0], cmap='plasma')
    plt.savefig("depth/block-insertion-0-end-episode-cam2_depth_estimated_nyu.png")

    
    print(np.shape(predicted_depth_begin_0))
    depth_estimated_0=np.concatenate((predicted_depth_begin_0[0],predicted_depth_begin_1[0],predicted_depth_begin_2[0]),axis=0)
    print(np.shape(depth_estimated_0))
    depth_estimated_1=np.concatenate((predicted_depth_end_0[0],predicted_depth_end_1[0],predicted_depth_end_2[0]),axis=0)
    print(np.shape(depth_estimated_1))
    obs_0 = {'color': color[0],'depth': depth_estimated_0} 
    obs_1 = {'color': color[-1],'depth': depth_estimated_1} 
    print(np.shape(color[0][0]))
    # obs_0 = {'color': color[0],'depth': predicted_depth_begin_0[0]} 
    # obs_1 = {'color': color[1],'depth': predicted_depth_end_0[0]} 
    cmap_0, hmap_0 = utils.get_fused_heightmap(obs_0, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    cmap_1, hmap_1 = utils.get_fused_heightmap(obs_1, cameras.RealSenseD415.CONFIG, np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]]), 0.003125)
    print(np.shape(cmap_0))
    print(np.shape(hmap_0))
    print(np.shape(cmap_1))
    print(np.shape(hmap_1))

    print(f'hmap_0 max: {np.max(hmap_0)}, hmap_0 min: {np.min(hmap_0)}, hmap_0 sum: {np.sum(hmap_0)}, hmap_0 type: {type(hmap_0)}, hmap_0 shape: {np.shape(hmap_0)} \n {hmap_0}')

    img = Image.fromarray(cmap_0, 'RGB')
    img.save('depth/begin_episode_homographie.png')
    img.show()
    img = Image.fromarray(cmap_1, 'RGB')
    img.save('depth/end_episode_homographie.png')
    img.show()

    plt.imshow(cmap_0)
    plt.savefig('depth/begin_episode_homographie.png')
    plt.imshow(cmap_1)
    plt.savefig('depth/end_episode_homographie.png')

    plt.imshow(hmap_0, cmap='plasma')
    plt.savefig('depth/begin_episode_homographie_depth_nyu.png')
    plt.imshow(hmap_1, cmap='plasma')
    plt.savefig('depth/end_episode_homographie_depth_nyu.png')