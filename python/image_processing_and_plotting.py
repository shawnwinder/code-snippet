''' snippet 1
plot multi-figures with matplotlib + boundingbox
'''
# for i in range(8):
#     plt.figure(i)
#     plt.title('image {}'.format(i + 1))

#     rect = patches.Rectangle((10,10), 50, 50,linewidth=1,
#     edgecolor='r',facecolor='none')
#     current_axis = plt.gca()
#     current_axis.add_patch(rect)
#     plt.imshow(np.ones((224, 224, 3)))


''' snippet 2
plot multi-subfigure in one figure with matplotlib + boundingbox
'''
# fig = plt.figure(figsize=(16,8))
# for i in range(8):
#     ax = plt.subplot(2,4,i + 1)
#     plt.title('image {}'.format(i + 1))
#     rect = patches.Rectangle((10,10), 50, 50,linewidth=1,
#     edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     plt.imshow(np.ones((224, 224, 3)))


''' snippet 3
preprocess lmdb image readed by "ImageInputOp"
'''
# # CHW -> HWC
# image = data[idx].transpose(1, 2, 0)
# # BGR -> RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # rescale image to float image, then it can be show by plt.imshow or cv2.imshow
# image = image / 255.0


''' snippet 4
plot graph for accumulation statistics like accuracy-epoch, loss-epoch
'''
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# def plot_history(epoch_results, name):
#     plt.figure(figsize=(12, 8))
#     # we can also add axis annotation here
#     plt.title(name)
#     plt.plot(range(1, 1 + len(epoch_results)), epoch_results)
#     plt.draw()
#     file_path = os.path.join(root_dir, '{}.png'.format(name))
#     plt.savefig(file_path)


''' snippet 5
plot graph for accumulation statistics like accuracy-epoch, loss-epoch
'''
# from caffe2.python import net_drawer
# mamc_training_graph = net_drawer.GetPydotGraph(
#     training_model.net.Proto().op,
#     "mamc_training",
#     rankdir="TB",)
# mamc_training_graph.write_svg("mamc_training_graph.svg")
#
# mamc_training_graph_mini = net_drawer.GetPydotGraphMinimal(
#     training_model.net.Proto().op,
#     "mamc_training_minimal",
#     rankdir="TB",
#     minimal_dependency=True)
# mamc_training_graph_mini.write_svg("mamc_training_graph_mini.svg")
# print("write graph over...")


''' snippet 6
plot multiple curves in one graph, for one-stage
'''
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import numpy as np
# import os
#
#
# def f1(x):
#     return 3*x + 4
#
# def f2(x):
#     return x**2
#
# root_dir = ''
# plt.figure(num=1, figsize=(12, 8))
#
# # we can also add axis annotation here
# X = np.arange(1, 10, 0.5)
# name = 'hello'
#
# plt.title(name)
# plt.plot(X, f1(X), 'b+-', lw=2, label='training_acc')
# plt.plot(X, f2(X), 'g*-', lw=2, label='validation_acc')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
#
# plt.draw()
# file_path = os.path.join(root_dir, '{}.png'.format(name))
# plt.savefig(file_path)



''' snippet 7
plot multiple curves in one graph, for multi-stage
'''
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# import numpy as np
# import os
#
#
# def f1(x):
#     return 3*x + 4
#
# def f2(x):
#     return x**2
#
# root_dir = ''
# X = np.arange(1, 10, 0.5)
# Y = np.arange(2, 11, 0.5)
#
# fig1 = 'acc'
# fig2 = 'acc5'
#
# plt.figure(fig1, figsize=(12, 8))
# plt.title('fig1')
# plt.xlabel('x1')
# plt.ylabel('y1')
#
# plt.figure(fig2, figsize=(12, 8))
# plt.title('fig2')
# plt.xlabel('x2')
# plt.ylabel('y2')
#
# plt.figure(fig1)
# plt.plot(X, f1(X), 'b.--', label='f1')
# plt.legend()
# plt.draw()
# file_path = os.path.join(root_dir, 'fig1.png')
# plt.savefig(file_path)
#
# plt.figure(fig2)
# plt.plot(Y, f1(Y), 'b.--', label='f1')
# plt.legend()
# plt.draw()
# file_path = os.path.join(root_dir, 'fig2.png')
# plt.savefig(file_path)
#
# plt.figure(fig1)
# plt.plot(X, f2(X), 'r.--', label='f2')
# plt.legend()
# plt.draw()
# file_path = os.path.join(root_dir, 'fig1.png')
# plt.savefig(file_path)
#
# plt.figure(fig2)
# plt.plot(Y, f2(Y), 'r.--', label='f2')
# plt.legend()
# plt.draw()
# file_path = os.path.join(root_dir, 'fig2.png')
# plt.savefig(file_path)



# snippet 8
# image rescale
'''
def rescale(image, SCALE):
    H, W, = image.shape[:-1]
    aspect = float(W) / H
    if aspect > 1:
        w = int(aspect * SCALE)
        img_scaled = cv2.resize(image, (w, SCALE))  # size=(W, H) in opencv
    elif aspect < 1:
        h = int(SCALE / aspect)
        img_scaled = cv2.resize(image, (SCALE, h))
    else:
        img_scaled = cv2.resize(image, (SCALE, SCALE))
    return img_scaled
'''

# snippet 9
# image central crop, channel normalization, color jitter
'''
def central_crop(image, CROP):
    h, w, c = image.shape
    assert(c == 3)
    h_beg = (h - CROP) // 2
    w_beg = (w - CROP) // 2
    return image[h_beg:(h_beg+CROP), w_beg:(w_beg+CROP), :]

def normalize_channel(image, MEAN, STD):
    return (image - MEAN) / STD

def color_jitter(img, JITTER):
    if JITTER:
        # img = cv2.cvtColor(image,  cv2.COLOR_BGR2RGB)  # cv2 defaul color code is BGR
        h, w, c = img.shape
        noise = np.random.randint(0, 50, (h, w)) # design jitter/noise here
        zitter = np.zeros_like(img)
        zitter[:,:,1] = noise
        noise_added = cv2.add(img, zitter)

        combined = np.vstack((img[:h,:,:], noise_added[h:,:,:]))
        return combined
    else:
        return img
'''
