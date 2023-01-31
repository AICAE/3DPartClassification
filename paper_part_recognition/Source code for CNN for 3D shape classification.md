## Shapenet for ModelNet dataset

Shapenet use ModelNet10 dataset, creator of dataset ModelNet

Leaderboard for modelnet

刚好最近看到一篇文章3D ShapeNets: A Deep Representation for Volumetric Shapes说的是将带深度信息的图像转化为3D体素。



## **Multi-view Convolutional Neural Networks for 3D Shape Recognition**

![img](http://vis-www.cs.umass.edu/mvcnn/images/mvcnn.png)



#### mvcnn  original impl

https://github.com/suhangpro/mvcnn

Matlab has various impl,  almost all ML toolkits

https://github.com/WeiTang114/MVCNN-TensorFlow  no preprocessor to gen views



#### Multiview pooling:

```py
#https://github.com/WeiTang114/MVCNN-TensorFlow/blob/master/model.py
def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_max(vp, [0], name=name)
    return vp
```



```py
tf.keras.layers.concatenate(view_pool, axis=0)


```



https://github.com/jongchyisu/mvcnn_pytorch

[Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)

class SVCNN(Model): Alex, VGG11 VGG16, RESNET

https://github.com/ace19-dev/mvcnn-tf/blob/master/nets/resnet_v2.py

#### mvcnn-tf

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.
Typical use:    ` from tensorflow.contrib.slim.nets import resnet_v2`

https://github.com/ace19-dev/mvcnn-tf



#### mvcnn-keras

https://github.com/Mannix1994/MVCNN-Keras

- modelnet40v1 (12 views w/ upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar) (204M)
- modelnet40v2 (80 views w/o upright assumption): [tarball](http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v2.tar) (1.3G)

### Blender script to generate views

[Blender script for rendering shaded images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)  not working links
[Blender script for rendering depth images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_depth.blend)

https://github.com/weiaicunzai/blender_shapenet_render





### MVSNet: Depth Inference for Unstructured Multi-view Stereo

https://github.com/YoYo000/MVSNet



### Multi-view Discrimination and Pairwise CNN (MDPCNN)  for View-based 3D Object Retrieval

Use only 3 views!

Zernike moments ([5]), Elevation Descriptors (EDs), LightField Descriptors (LFDs) ([6]), Bag of Visual Features (BoVF) ([7]), Histogram of Orientation Gradient (HOG) ([8], [9]), Compact Multi-view Descriptors (CMVDs) ([10]), and Multiview and Multivariate Gaussian descriptor (MMG) ([11]).

A. Clustering

B. Pairwise Sample Generation

C. Multi-Batch


###  4 Datasets that are too small for each category

• ETH 3D object dataset ([38]), in this dataset, there are 80 objects that belong to 8 categories, and 41 different view images are used to represent each object from ETH dataset. In total, it contains 3280 views belonging to 8 categories.

• NTU60 3D model dataset ([6]), in this dataset, there are 549 objects that belong to 47 categories, and 60 different view samples are utilized to describe each object from NTU60 dataset. In total, it contains 32940 views belonging to 47 categories.

• MVRED 3D category dataset ([39]), in this dataset, there are 505 objects that belongs to 61 categories, and 36 different view images are included in each object from MVRED dataset. In total, it contains 18180 views belonging to 61 categories

### Harmonized Bilinear

Multi-view Harmonized Bilinear Network for 3D Object Recognition



OrthographicNet: A Deep Transfer Learning Approach for 3D Object Recognition in Open-Ended Domains

but point cloud, not image data. still depth-map

## PointMVSNet :  from 2D image to 3D point cloud

https://github.com/callmeray/PointMVSNet   from 2D image to 3D point cloud

https://github.com/garyli1019/pointnet-keras

### PVNet: A Joint Convolutional Network of Point Cloud and Multi-View for 3D Shape Recognition



## Voxnet

https://github.com/dimatura/voxnet

32X32X32 binary voxel 3D matrix

https://github.com/tobiagru/Deep-3D-Obj-Recognition              keras

### FusionNet: 3D Object Classification Using Multiple Data Representations



