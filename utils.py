import string
import numpy as np
from skimage.transform import resize
import os.path as osp
import random
import math
from skimage.morphology import disk
from skimage.filters import rank
from skimage.transform import resize
import pickle
from multiprocessing import Process, Queue, Pool, Lock
import traceback
import copy
import PIL.Image as Image
import yaml
import time
import md5
import matplotlib.pyplot as plt

def norm_image(image):
	mean = [103.939, 116.779, 123.68]
	image = image[..., ::-1]
	image[..., 0] -= mean[0]
	image[..., 1] -= mean[1]
	image[..., 2] -= mean[2]
	return image
	
def denorm_image(image):
	mean = [103.939, 116.779, 123.68]
	image[..., 0] += mean[0]
	image[..., 1] += mean[1]
	image[..., 2] += mean[2]
	image = image[..., ::-1]
	return image

class ParallelModel(KM.Model):
    """Adapted from Mask RCNN of matterport. Model wrapper to fit in multi-gpus.
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/parallel_model.py
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged

"""
util
"""

debug_mode = False
def cprint(string, style = None):
    if not debug_mode and style != bcolors.FAIL and style != bcolors.OKBLUE:
        return
    if style is None:
        print(str(string))
    else:
        print(style + str(string) + bcolors.ENDC)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_img(img_path):
    cprint('Reading Image ' + img_path, bcolors.OKGREEN)
    uint_image = np.array(Image.open(img_path))
    if len(uint_image.shape) == 2:
        tmp_image = np.zeros(uint_image.shape + (3,), dtype=np.uint8)
        tmp_image[:,:,0] = tmp_image[:,:,1] = tmp_image[:,:,2] = uint_image
        uint_image = tmp_image
    return np.array(uint_image, dtype=np.float32)/255.0
        
def read_mask(mask_path):
    #read mask
    m_uint = np.array(Image.open(mask_path))
    fg = np.unique(m_uint)
    if not (len(m_uint.shape) == 2 and ((len(fg) == 2 and fg[0] == 0 and fg[1] == 255) or (len(fg) == 1 and (fg[0] == 0 or fg[0] == 255)))):
        print(mask_path, fg, m_uint.shape)
        raise Exception('Error in reading mask')
    return np.array(m_uint, dtype=np.float32) / 255.0
    
    
def read_flo_file(file_path):
    """
    reads a flo file, it is for little endian architectures,
    first slice, i.e. data2D[:,:,0], is horizontal displacements
    second slice, i.e. data2D[:,:,1], is vertical displacements

    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic[0]:
            cprint('Magic number incorrect. Invalid .flo file: %s' % file_path, bcolors.FAIL)
            raise  Exception('Magic incorrect: %s !' % file_path)
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            data2D = np.reshape(data, (h[0], w[0], 2), order='C')
            return data2D


def write_flo_file(file_path, data2D):
    """
    writes a flo file, it is for little endian architectures,
    first slice, i.e. data2D[:,:,0], is horizontal displacements
    second slice, i.e. data2D[:,:,1], is vertical displacements

    """
    with open(file_path, 'wb') as f:
        magic = np.array(202021.25, dtype='float32')
        magic.tofile(f)
        h = np.array(data2D.shape[0], dtype='int32')
        w = np.array(data2D.shape[1], dtype='int32')
        w.tofile(f)
        h.tofile(f)
        data2D.astype('float32').tofile(f);
        
def add_noise_to_mask(cmask, r_param = (15, 15), mult_param = (20, 5), threshold = .2):
    radius = max(np.random.normal(*r_param), 1)
    mult = max(np.random.normal(*mult_param), 2)
    
    selem = disk(radius)
    mask2d = np.zeros(cmask.shape + (2,))
    mask2d[:, :, 0] = rank.mean((1 - cmask).copy(), selem=selem) / 255.0
    mask2d[:, :, 1] = rank.mean(cmask.copy(), selem=selem) / 255.0

    exp_fmask = np.exp(mult * mask2d);
    max_fmask = exp_fmask[:,:,1] / np.sum(exp_fmask, 2);
    max_fmask[max_fmask < threshold] = 0;
    
    return max_fmask
    
class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def parse_file(input_path, output_path, dictionary):
    with open(input_path, 'r') as in_file:
        with open(output_path, 'w') as out_file:
            data = string.Template(in_file.read())
            out_file.write(data.substitute(**dictionary))
            

def crop(img, bbox, output_shape = None, resize_order = 1, clip = True):
    bsize = bbox.size()
    if bsize[0] == 0:
        raise Exception('Cropping bbox can not be empty.')
    
    img_bbox = BBox(0, img.shape[0], 0, img.shape[1])
    intbox = img_bbox.copy()
    intbox.intersect(bbox)
    
    output = np.zeros(bsize + img.shape[2:])
    output[(intbox.top-bbox.top):(intbox.bottom-bbox.top), (intbox.top-bbox.left):(intbox.bottom-bbox.left)] = img[intbox.top:intbox.bottom, intbox.left:intbox.right]
    
    if output_shape is None or tuple(output_shape) == intbox.size():
        return output
    return resize(output, output_shape, order = resize_order, mode = 'nearest', clip = clip, preserve_range=True)
        

def crop_undo(cropped_img, cropping_bbox, img_shape, resize_order = 1):
    bsize = cropping_bbox.size()
    if bsize != cropped_img.shape[:2]:
        cropped_img = resize(cropped_img, bsize, order = resize_order, mode = 'nearest', preserve_range=True)
    img = np.zeros(img_shape + cropped_img.shape[2:])    
    img_bbox = BBox(0, img.shape[0], 0, img.shape[1])
    intbox = img_bbox.copy()
    intbox.intersect(cropping_bbox)
    
    img[intbox.top:intbox.bottom, intbox.left:intbox.right] = cropped_img[(intbox.top-cropping_bbox.top):(intbox.bottom-cropping_bbox.top), (intbox.top-cropping_bbox.left):(intbox.bottom-cropping_bbox.left)]
    return img

def change_coordinates(array, down_scale, offset, order = 0, preserve_range = True):
    ##in caffe we have label_cordinate = down_scale * 'name'_coordinate + offset
    ##in python resize we have label_coordinate = down_scale * 'name'_coordinate + (down_scale - 1)/2 - pad
    ## ==> (down_scale - 1)/2 - pad = offset ==> pad = -offset + (down_scale - 1)/2
    pad = int(-offset + (down_scale - 1)/2)
    orig_h = array.shape[0]
    orig_w = array.shape[1]
    new_h = int(np.ceil(float(orig_h + 2 * pad) / down_scale))
    new_w = int(np.ceil(float(orig_w + 2 * pad) / down_scale))
    #floor or ceil?
    if pad > 0:
        pad_array = ((0,0),) * (len(array.shape) - 2) + ((pad, int(new_h * down_scale - orig_h - pad)), (pad, int(new_w * down_scale - orig_w - pad)))
        new_array = np.pad(array, pad_array, 'constant')
    elif pad == 0:
        new_array = array
    else:
        raise Exception
        
    if new_h != orig_h or new_w != orig_w:
        return resize(new_array, (new_h,new_w) + array.shape[2:], order = order, preserve_range = preserve_range)
    else:
        return new_array.copy()
        
#defaults is a list of (key, val) is val is None key is required field
def check_params(params, **kwargs):
    for key, val in kwargs.items():
        key_defined = (key in params.keys())
        if val is None:
            assert key_defined, 'Params must include {}'.format(key)
        elif not key_defined:
            params[key] = val

def load_netflow_db(annotations_file, split, shuffle = False):
    if split == 'training':
        split = 1
    if split == 'test':
        split = 2
    annotations = np.loadtxt(annotations_file)
    frame_indices = np.arange(len(annotations))
    frame_indices = frame_indices[ annotations == split ]
    length = len(frame_indices)
    data_dir = osp.join(osp.dirname(osp.abspath(annotations_file)), 'data/')
    if shuffle:
        random.shuffle( frame_indices)

    return dict(frame_indices=frame_indices, data_dir=data_dir, length=length)

def read_netflow_instance(netflow_db, instance_id):
    data_dir = netflow_db['data_dir']
    instance_id = netflow_db['frame_indices'][instance_id]
    instance_id = instance_id + 1
    img1 = np.array(Image.open(osp.join(data_dir, '%05d_img1.ppm' % instance_id)).astype(np.float32)) / 255.0
    img2 = np.array(Image.open(osp.join(data_dir, '%05d_img2.ppm' % instance_id)).astype(np.float32)) / 255.0
    flow = read_flo_file( osp.join(data_dir, '%05d_flow.flo' % instance_id))
    return img1, img2, flow


#def compute_flow(T1, T2, object_size, img_size, flow = None):
        #newx = np.arange(img_size[1])
        #newy = np.arange(img_size[0])
        #mesh_grid = np.stack(np.meshgrid(newx, newy), axis = 0)
        #locs1 = np.array(mesh_grid, dtype='float')
        #locs2 = np.array(locs1.copy(), dtype='float')
        #if flow is not None:
             #locs2 += flow.transpose((2,0,1))
        #x,y = T1.transform_points(locs1[0].ravel(), locs1[1].ravel(), locs1[0].shape)
        #locs1 = np.concatenate((x,y)).reshape((2,) + img_size)
        #x,y = T2.transform_points(locs2[0].ravel(), locs2[1].ravel(), locs2[0].shape)
        #locs2 = np.concatenate((x,y)).reshape((2,) + img_size)
        #flow_trans = locs2 - locs1
        
        #final_flow = np.zeros((2,) + img_size)
        #T1_cp = copy.deepcopy(T1)
        #T1_cp.color_adjustment_param = None
        #final_flow[0] = T1_cp.transform_img(flow_trans[0], object_size, flow_trans[0].shape, cval=0)
        #final_flow[1] = T1_cp.transform_img(flow_trans[1], object_size, flow_trans[1].shape, cval=0)
        #return final_flow.transpose((1,2,0))


def compute_flow(T1, T2, object_size, img_size, flow = None):
    assert len(img_size) == 2 and len(object_size) == 2
    #final_flow(T1(i,j)) = T2( (i,j) + f1(i,j) ) - T1(i,j)
    # = (T2(i,j) + T2(f1(i,j)) - T2((0,0))) - T1(i, j)
    # = T2(i,j) - T1(i,j) + T2(f1(i,j)) - T2((0,0))
    # = A + B - T2((0,0)) where A = T2(i,j) - T1(i,j), B = T2(f1(i,j))
    # A(T1(i, j)) = T2(i,j) - T1(i,j) ==> A((m,n)) = T2(T1^-1(m,n)) - (m, n)
    # B(T1(i, j)) = T2(f1(i,j)) ==(see *)==> BT1(m,n) = B(T^-1(m,n))
        
    # * Given an image I and transformation T: (IT is the image after applying transformation T)
    # IT[k,l] = I[T^-1(k,l)]
        
    # 1) Compute A
    newx = np.arange(img_size[1])
    newy = np.arange(img_size[0])
    mesh_grid = np.stack(np.meshgrid(newx, newy), axis = 0)
    locs1 = np.array(mesh_grid, dtype='float')
    x,y = T1.itransform_points(locs1[0].ravel(), locs1[1].ravel(), object_size)
    x,y = T2.transform_points(x, y, object_size)
    locs2 = np.concatenate((x,y)).reshape((2,) + locs1[0].shape)
    final_flow = locs2 - locs1
        
    # 2) Compute B - T2((0,0))
    if flow is not None:
        # B
        x,y = T2.transform_points(flow[:,:,0].ravel(), flow[:,:,1].ravel(), object_size)
        b_flow = np.concatenate((x,y)).reshape((2,) + img_size)
        T1_cp = copy.deepcopy(T1)
        T1_cp.color_adjustment_param = None
        b_flow[0] = T1_cp.transform_img(b_flow[0], object_size, b_flow[0].shape, cval=0)
        b_flow[1] = T1_cp.transform_img(b_flow[1], object_size, b_flow[1].shape, cval=0)
        #T2((0,0))
        x0, y0 = T2.transform_points(np.array((0,)), np.array((0,)), object_size)
        b_flow[0] -= x0[0]
        b_flow[1] -= y0[0]
        
        #Add it to the final flow
        final_flow += b_flow
    return final_flow.transpose((1,2,0))
        
def sample_trans(base_tran, trans_dist):
    if base_tran is None and trans_dist is None:
        return None
    if base_tran is None:
        return trans_dist.sample()
    if trans_dist is None:
        return base_tran
    return base_tran + trans_dist.sample()

#Not Tested:
#def sample_trans(base_tran, *args):
    #cur_trans = base_tran
    #for tran_dist in args:
        #new_trans = None
        #if tran_dist is not None:
            #new_trans = trans_dist.sample()
        
        #if cur_trans is None:
            #cur_trans = new_trans
        #elif new_trans is not None:
            #cur_trans = cur_trans + tran.sample()
    #return cur_trans
#################################### Util classes
#Integer value bbox 
#bottom and right are exlusive
class BBox:
    def __init__(self, top, bottom, left, right):
        self.init(top, bottom, left, right)
           
    def init(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def intersect(self, bbox):
        if self.isempty() or bbox.isempty():
            self.init(0,0,0,0)
            return
        self.top = max(self.top, bbox.top)
        self.bottom = min(self.bottom, bbox.bottom)
        self.left = max(self.left, bbox.left)
        self.right = min(self.right, bbox.right)
   
    def pad(self, rpad, cpad=None):
        if self.isempty():
            raise Exception('Can not pad empty bbox')
        if cpad is None:
            cpad = rpad
        self.top -= rpad
        self.bottom += rpad
        self.left -= cpad
        self.right += cpad
       
    def scale(self, rscale, cscale=None):
        if self.isempty():
            return
        if cscale is None:
            cscale = rscale
        rpad = int((rscale - 1) * (self.bottom - self.top) / 2.0)
        cpad = int((cscale - 1) * (self.right - self.left) / 2.0)
        self.pad(rpad, cpad)
    
    def move(self, rd, cd):
        self.top += rd
        self.bottom += rd
        self.left += cd
        self.right += cd
        
    def isempty(self):
        return (self.bottom <= self.top) or (self.right <= self.left)
    
    def size(self):
        if self.isempty():
            return (0,0)
        return (self.bottom - self.top, self.right - self.left)
        
    def copy(self):
        return copy.copy(self)
    @staticmethod
    def get_bbox(img):
        if img.sum() == 0:
            return BBox(0, 0, 0, 0)
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
        return BBox(top, bottom+1, left, right+1)
    
class Cache:
    def __init__(self, max_size = 10):
        self.max_size = max_size
        self.cache = dict()
        self.key_queue = []
    def has_key(self, key):
        return self.cache.has_key(key)

    def __setitem__(self, key, value):
        if self.cache.has_key(key):
            self.__delitem__(key)
        self.cache.__setitem__(key, copy.deepcopy(value))
        self.key_queue.append(key)
        if len(self.cache) > self.max_size:
            self.__delitem__(self.key_queue[0])
        
    def __getitem__(self, key):
        assert self.cache.has_key(key)
        self.key_queue.remove(key)
        self.key_queue.append(key)
        return copy.deepcopy(self.cache.__getitem__(key))
        
    def __delitem__(self, key):
        self.cache.__delitem__(key)
        self.key_queue.remove(key)

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
        
########################################################################### Sequence Generator ######################################################################## 
class VideoPlayer:
    def __init__(self, video_item, base_trans = None, frame_trans_dist = None, frame_noise_dist = None, step = 1, offset = 0, max_len = np.inf, flo_method = None):
        self.cache = Cache()
        self.name = video_item.name
        if offset != 0:
            self.name += '_o' + str(offset)
            
        if step != 1:
            self.name += '_s' + str(step)
            
        if not np.isinf(max_len):
            self.name += '_m' + str(max_len)
        
        self.video_item = video_item
        self.step = step
        self.offset = offset
        self.max_len = max_len
        self.flo_method = flo_method
        
        ##compute img_ids
        if step > 0:
            a = self.offset
            b = self.video_item.length
        elif step < 0:
            a = self.video_item.length - 1 - self.offset
            b = - 1    
        self.img_ids = range(a, b, self.step)
        if not np.isinf(self.max_len):
            self.img_ids = self.img_ids[:self.max_len]
        self.length = len(self.img_ids)
        #cprint(str(self.img_ids), bcolors.OKBLUE)
        ##compute mappings
        self.mappings = None
        self.gt_mappings = None
        if base_trans is not None or frame_trans_dist is not None:
            self.name += '_' + str(random.randint(0, 1e10))
            self.mappings = []
            self.gt_mappings = []
            for i in range(self.length):
                mapping = sample_trans(base_trans, frame_trans_dist)
                gt_mapping = sample_trans(mapping, frame_noise_dist)
                self.mappings.append(mapping)
                self.gt_mappings.append(gt_mapping)
    
    def get_frame(self, frame_id, compute_iflow = False):
        if self.cache.has_key(frame_id):
            img, mask, obj_size = self.cache[frame_id]
            assert(np.all(img >= 0) and np.all(img <= 1.0))
        else:
            img_id = self.img_ids[frame_id]
            img = self.video_item.read_img(img_id)
            assert(np.all(img >= 0) and np.all(img <= 1.0))
            try:
                mask = self.video_item.read_mask(img_id)
                obj_size = np.array(BBox.get_bbox(mask).size())
            except IOError:
                cprint('Failed to load mask \'' + str(img_id) + '\' for video \'' + self.name + '\'. Return None mask..', bcolors.FAIL)
                mask = None
                obj_size = np.array([50, 50])
                
            if self.mappings is not None:
                img = self.mappings[frame_id].transform_img(img.copy(), obj_size, img.shape[:2], mask)
                if mask is not None:
                    mask  = self.mappings[frame_id].transform_mask(mask.copy(), obj_size, mask.shape)[0]
                    mask[mask == -1] = 0
            self.cache[frame_id] = (img, mask, obj_size)
        output = dict(image=img, mask=mask)
        
        if compute_iflow:
            try:
                iflow = self.video_item.read_iflow(img_id, self.step, self.flo_method)
            except Exception as e:
                cprint('Failed to load \'' + self.flo_method + '\' iflow for video ' + self.name + '. Return zero iflow..', bcolors.FAIL)
                iflow = np.zeros(img.shape[:2] + (2,))
            if self.mappings is None:
                output['iflow'] = iflow
            else:
                output['iflow'] = compute_flow(self.mappings[frame_id], self.gt_mappings[frame_id - 1], obj_size, img.shape[:2], flow = iflow)
        
        if output.has_key('mask') and output['mask'] is not None:
            assert output['mask'].shape[0] == output['image'].shape[0] and output['mask'].shape[1] == output['image'].shape[1]
        if output.has_key('iflow'):
            assert output['iflow'].shape[0] == output['image'].shape[0] and output['iflow'].shape[1] == output['image'].shape[1]
        return output

class ImagePlayer:
    def __init__(self, image_item, base_trans, frame_trans_dist, frame_noise_dist, compute_iflow = False, length = 2):
        self.name = image_item.name + '_' + str(random.randint(0, 1e10))
        self.length = length
        self.imgs = []
        self.masks = []
        self.image_item = image_item
        
        img = image_item.read_img()
        mask = image_item.read_mask()
        obj_size = BBox.get_bbox(mask).size()
        mappings = []
        gt_mappings = []
        for i in range(length):
            mapping = sample_trans(base_trans, frame_trans_dist)
            #debug
            #print '>'*10, 'Base Trans = ', base_trans
            #print '>'*10, 'Frame Trans = ', frame_trans_dist
            gt_mapping = sample_trans(mapping, frame_noise_dist)
            if gt_mapping is not None:
                timg = mapping.transform_img(img.copy(), obj_size, img.shape[:2], mask)
                tmask  = mapping.transform_mask(mask.copy(), obj_size, mask.shape)[0]
            else:
                timg = img.copy()
                tmask = mask.copy()
            tmask[tmask == -1] = 0          
            self.imgs.append(timg)
            self.masks.append(tmask)
            gt_mappings.append(gt_mapping)
            mappings.append(mapping)
            
        if compute_iflow:
            self.iflows = [None]
            for i in range(1, length):
                iflow = compute_flow(mappings[i], gt_mappings[i - 1], obj_size, mask.shape)
                self.iflows.append(iflow)

    def get_frame(self, frame_id, compute_iflow = False):
        output = dict(image=self.imgs[frame_id], mask=self.masks[frame_id])
        if compute_iflow:
            output['iflow'] = self.iflows[frame_id]
        return output 

########################################################################### Read DBs into DBItems ################################################################################
class DAVIS:
    def __init__(self, cfg):
        self.cfg = cfg
    # DAVIS: 1376 Test, 2079 Training
    # Jump-Cut: ?
    def getItems(self, sets, categories = None):
        if isinstance(sets, basestring):
            sets = [sets]
        if isinstance(categories, basestring):
            categories = [categories]
        if len(sets) == 0:
            return []
        with open(self.cfg['DB_INFO'],'r') as f:
            db_info = yaml.load(f)
        sequences = [x for x in db_info['sequences'] if x['set'] in sets and (categories is None or x['name'] in categories)]
        assert len(sequences) > 0
        
        items = []
        for seq in sequences:
            name = seq['name']
            img_root = osp.join(self.cfg['SEQUENCES_DIR'], name)
            ann_root = osp.join(self.cfg['ANNOTATION_DIR'], name)
            item = DBDAVISItem(name, img_root, ann_root, seq['num_frames'])
            items.append(item)
        return items

class COCO:
    def __init__(self, db_path, dataType):
        self.pycocotools = __import__('pycocotools.coco')
        if dataType == 'training':
            dataType = 'train2014'
        elif dataType == 'test':
            dataType = 'val2014'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')
        
        self.db_path = db_path
        self.dataType = dataType
        
    def getItems(self, cats=[], areaRng=[], iscrowd=False):
        
        annFile='%s/annotations/instances_%s.json' % (self.db_path, self.dataType)
        
        coco = self.pycocotools.coco.COCO(annFile)
        catIds = coco.getCatIds(catNms=cats);
        anns = coco.getAnnIds(catIds=catIds, areaRng=areaRng, iscrowd=iscrowd)
        cprint(str(len(anns)) + ' annotations read from coco', bcolors.OKGREEN)
        
        items = []
        for i in range(len(anns)):
            ann = anns[i]
            item = DBCOCOItem('coco-'  + self.dataType + str(i), self.db_path, self.dataType, ann, coco, self.pycocotools)
            items.append(item)
        return items    
        
class PASCAL_READ_MODES:
    #Returns list of DBImageItem each has the image and one object instance in the mask
    INSTANCE = 0
    #Returns list of DBImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    #Returns list of DBImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2
class PASCAL:
    def __init__(self, db_path, dataType):
        if dataType == 'training':
            dataType = 'train'
        elif dataType == 'test':
            dataType = 'val'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')
        
        self.db_path = db_path
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        self.name_id_map = dict(zip(classes, range(1, len(classes) + 1)))
        self.id_name_map = dict(zip(range(1, len(classes) + 1), classes))
        self.dataType = dataType
    
      
    def getCatIds(self, catNms=[]):
        return [self.name_id_map[catNm] for catNm in catNms]
    
    def get_anns_path(self, read_mode):
        return osp.join(self.db_path, self.dataType + '_' + str(read_mode) + '_anns.pkl') 
    
    def get_unique_ids(self, mask, return_counts=False, exclude_ids = [0, 255]):
        ids, sizes = np.unique(mask, return_counts=True)
        ids = list(ids)
        sizes = list(sizes)
        for ex_id in exclude_ids:
          if ex_id in ids:
            id_index = ids.index(ex_id)
            ids.remove(ex_id)
            sizes.remove(sizes[id_index])
        
        assert(len(ids) == len(sizes))
        if return_counts:
            return ids, sizes
        else:
            return ids
    
    def create_anns(self, read_mode):
        with open(osp.join(self.db_path, 'ImageSets', 'Segmentation', self.dataType + '.txt'), 'r') as f:
            lines = f.readlines()    
            names = []
            for line in lines:
                if line.endswith('\n'):
                    line = line[:-1]
                if len(line) > 0:
                    names.append(line)
        anns = []
        for item in names:
            mclass_path = osp.join(self.db_path, 'SegmentationClass', item + '.png')
            mobj_path = osp.join(self.db_path, 'SegmentationObject', item + '.png')
            mclass_uint = np.array(Image.open(mclass_path))
            mobj_uint = np.array(Image.open(mobj_path))
            class_ids = self.get_unique_ids(mclass_uint)
            obj_ids, obj_sizes = self.get_unique_ids(mobj_uint, return_counts = True)

            if read_mode == PASCAL_READ_MODES.INSTANCE:  
                for obj_idx in xrange(len(obj_ids)):
                    class_id = int(np.median(mclass_uint[mobj_uint == obj_ids[obj_idx]]))
                    assert( class_id != 0 and class_id != 255 and obj_ids[obj_idx] != 0 and obj_ids[obj_idx] != 255)
                    anns.append(dict(image_name=item, mask_name=item, object_ids=[obj_ids[obj_idx]], class_ids=[class_id], object_sizes = [obj_sizes[obj_idx]]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC:
                for class_id in class_ids:
                    assert(class_id != 0 or class_id != 255)
                    anns.append(dict(image_name=item, mask_name=item, class_ids=[class_id]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:  
                anns.append(dict(image_name=item, mask_name=item, class_ids=class_ids))
        with open(self.get_anns_path(read_mode), 'w') as f:
            pickle.dump(anns, f)
            
    def load_anns(self, read_mode):
        path = self.get_anns_path(read_mode)
        if not osp.exists(path):
            self.create_anns(read_mode)
        with open(path, 'rb') as f:
            anns = pickle.load(f)
        return anns
    
    def get_anns(self, catIds=[], areaRng=[], read_mode = PASCAL_READ_MODES.INSTANCE):
        if areaRng == []:
            areaRng = [0, np.inf]
        anns = self.load_anns(read_mode)
        if catIds == [] and areaRng == [0, np.inf]:
            return anns
        
        if read_mode == PASCAL_READ_MODES.INSTANCE:
            filtered_anns = [ann for ann in anns if ann['class_ids'][0] in catIds and areaRng[0] < ann['object_sizes'][0] and ann['object_sizes'][0] < areaRng[1]]
        else:
            filtered_anns = []
            catIds_set = set(catIds)
            for ann in anns:
                class_inter = set(ann['class_ids']) & catIds_set
                #remove class_ids that we did not asked for (i.e. are not catIds_set)
                if len(class_inter) > 0:
                    ann = ann.copy()
                    ann['class_ids'] = sorted(list(class_inter))
                    filtered_anns.append(ann)
        return filtered_anns
    
    def getItems(self, cats=[], areaRng=[], read_mode = PASCAL_READ_MODES.INSTANCE):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)
        
        anns = self.get_anns(catIds=catIds, areaRng=areaRng, read_mode=read_mode)
        cprint(str(len(anns)) + ' annotations read from pascal', bcolors.OKGREEN)
        
        items = []
        
        ids_map = None
        if read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
            old_ids = catIds
            new_ids = range(1, len(catIds) + 1)
            ids_map = dict(zip(old_ids, new_ids))
        for i in range(len(anns)):
            ann = anns[i]
            img_path = osp.join(self.db_path, 'JPEGImages', ann['image_name'] + '.jpg')
            if read_mode == PASCAL_READ_MODES.INSTANCE:
                mask_path = osp.join(self.db_path, 'SegmentationObject', ann['mask_name'] + '.png')
                item = DBPascalItem('pascal-'  + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path, mask_path, ann['object_ids'])
            else:
                mask_path = osp.join(self.db_path, 'SegmentationClass', ann['mask_name'] + '.png')
                item = DBPascalItem('pascal-'  + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path, mask_path, ann['class_ids'], ids_map)
            items.append(item)
        return items
    @staticmethod
    def cluster_items(items):
        clusters = {}
        for i, item in enumerate(items):
            assert(isinstance(item, DBPascalItem))
            item_id = item.obj_ids
            assert(len(item_id) == 1), 'For proper clustering, items should only have one id'
            item_id = item_id[0]
            if clusters.has_key(item_id):
                clusters[item_id].append(item)
            else:
              clusters[item_id] = DBImageSetItem('set class id = ' + str(item_id), [item])    
        return clusters
########################################################################### DB Items ###################################################################################
class DBVideoItem:
    def __init__(self, name, length):
        self.name = name
        self.length = length
    def read_img(self, img_id):
        pass
    def read_mask(self, img_id):
        pass

class DBDAVISItem(DBVideoItem):
    def __init__(self, name, img_root, ann_root, length):
        DBVideoItem.__init__(self, name, length)
        self.img_root = img_root
        self.ann_root = ann_root
    
    def read_img(self, img_id):
        file_name = osp.join(self.img_root, '%05d.jpg' % (img_id))
        return read_img(file_name)
        
    def read_mask(self, img_id):
        file_name = osp.join(self.ann_root, '%05d.png' % (img_id))
        mask = read_mask(file_name)
        return mask 
        
    def read_iflow(self, img_id, step, method):
        if method == 'LDOF':
            if step == 1:
                flow_name = osp.join(self.ann_root, '%05d_inv_LDOF.flo' % (img_id))
            elif step == -1:
                flow_name = osp.join(self.ann_root, '%05d_LDOF.flo' % (img_id))
            else:
                raise Exception('unsupported flow step for LDOF')
        elif method == 'EPIC':
            if step == 1:
                flow_name = osp.join(self.ann_root, '%05d_inv.flo' % (img_id))
            elif step == -1:
                flow_name = osp.join(self.ann_root, '%05d.flo' % (img_id))
            else:
                raise Exception('unsupported flow step for EPIC')
        else:
            raise Exception('unsupported flow algorithm')
        try:
            return read_flo_file(flow_name)
        except IOError as e:
            print "Unable to open file", str(e)#Does not exist OR no read permissions

class DBImageSetItem(DBVideoItem):
    def __init__(self, name, image_items = []):
        DBVideoItem.__init__(self, name, len(image_items))
        self.image_items = image_items
    def append(self, image_item):
        self.image_items.append(image_item)
        self.length += 1
    def read_img(self, img_id):
        return self.image_items[img_id].read_img()
        
    def read_mask(self, img_id):
        return self.image_items[img_id].read_mask()

#####
class DBImageItem:
    def __init__(self, name):
        self.name = name
    def read_mask(self):
        pass
    def read_img(self):
        pass

class DBCOCOItem(DBImageItem):
    def __init__(self, name, db_path, dataType, ann_info, coco_db, pycocotools):
        DBImageItem.__init__(self, name)
        self.ann_info = ann_info
        self.db_path = db_path
        self.dataType = dataType
        self.coco_db = coco_db
        self.pycocotools = pycocotools
        
    def read_mask(self):
        ann = self.coco_db.loadAnns(self.ann_info)[0]
        img_cur = self.coco_db.loadImgs(ann['image_id'])[0]
        
        rle = self.pycocotools.mask.frPyObjects(ann['segmentation'], img_cur['height'], img_cur['width'])
        m_uint = self.pycocotools.mask.decode(rle)
        m = np.array(m_uint[:, :, 0], dtype=np.float32)
        return m
    
    def read_img(self):
        ann = self.coco_db.loadAnns(self.ann_info)[0]
        img_cur = self.coco_db.loadImgs(ann['image_id'])[0]
        img_path = '%s/images/%s/%s' % (self.db_path, self.dataType, img_cur['file_name'])
        return read_img(img_path)
    
"""
ss_datalayer
"""

class DBPascalItem(DBImageItem):
    def __init__(self, name, img_path, mask_path, obj_ids, ids_map = None):
        DBImageItem.__init__(self, name)
        self.img_path = img_path
        self.mask_path = mask_path
        self.obj_ids = obj_ids
        if ids_map is None:
            self.ids_map = dict(zip(obj_ids, np.ones(len(obj_ids))))
        else:
            self.ids_map = ids_map
    def read_mask(self, orig_mask=False):
        mobj_uint = np.array(Image.open(self.mask_path))
        
        if orig_mask:
            return mobj_uint.astype(np.float32)
        m = np.zeros(mobj_uint.shape, dtype=np.float32)
        for obj_id in self.obj_ids:
            m[mobj_uint == obj_id] = self.ids_map[obj_id]
        #m[mobj_uint == 255] = 255
        return m
    def read_img(self):
        return read_img(self.img_path)

class DBInterface():
    def __init__(self, params):
        self.lock = Lock()
        self.params = params
        self.load_items()
        
        # initialize the random generator
        self.init_randget(params['read_mode'])
        self.cycle = 0
        
    def init_randget(self, read_mode):
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385) #>>>Do not change<<< Fixed seed for deterministic mode. 
    
    def update_seq_index(self):
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0
    
    def next_pair(self):
        with self.lock:
            end_of_cycle = self.params.has_key('db_cycle') and self.cycle >= self.params['db_cycle']
            if end_of_cycle:
                assert(self.params['db_cycle'] > 0)
                self.cycle = 0
                self.seq_index = len(self.db_items)
                self.init_randget(self.params['read_mode'])
                
            self.cycle += 1
            base_trans = None if self.params['image_base_trans'] is None else self.params['image_base_trans'].sample()
            self.update_seq_index()
            if self.params['output_type'] == 'single_image':
                db_item = self.db_items[self.seq_index]
                assert(isinstance(db_item, DBImageItem))
                player = ImagePlayer(db_item, base_trans, None, None, length = 1)
                return player, [0], None
            elif self.params['output_type'] == 'image_pair':
                imgset, second_index = self.db_items[self.seq_index]
                player = VideoPlayer(imgset, base_trans, self.params['image_frame_trans'])
                set_indices = range(second_index) + range(second_index+1, player.length)
                assert(len(set_indices) >= self.params['k_shot'])
                self.rand_gen.shuffle(set_indices)
                first_index = set_indices[:self.params['k_shot']]
                return player, first_index, second_index
            else:
                raise Exception('Only single_image and image_pair mode are supported')
    
    def _remove_small_objects(self, items):
        filtered_item = []
        for item in items:
            mask = item.read_mask()
            if change_coordinates(mask, 32.0, 0.0).sum() > 2:
                filtered_item.append(item)
        return filtered_item
    
    def load_items(self):
        self.db_items = []
        if self.params.has_key('image_sets'):
            for image_set in self.params['image_sets']:
                if image_set.startswith('pascal') or image_set.startswith('sbd'):
                    if image_set.startswith('pascal'):
                        pascal_db = PASCAL(self.params['pascal_path'], image_set[7:])   
                    elif image_set.startswith('sbd'):
                        pascal_db = PASCAL(self.params['sbd_path'], image_set[4:])   
                    #reads single image and all semantic classes are presented in the label
                    
                    if self.params['output_type'] == 'single_image':
                        items = pascal_db.getItems(self.params['pascal_cats'], self.params['areaRng'], read_mode = PASCAL_READ_MODES.SEMANTIC_ALL)
                    #reads pair of images from one semantic class and and with binary labels
                    elif self.params['output_type'] == 'image_pair':
                        items = pascal_db.getItems(self.params['pascal_cats'], self.params['areaRng'], read_mode = PASCAL_READ_MODES.SEMANTIC)
                        items = self._remove_small_objects(items)
                    else:
                        raise Exception('Only single_image and image_pair mode are supported')
                    self.db_items.extend(items)
                else:
                    raise Exception
            cprint('Total of ' + str(len(self.db_items)) + ' db items loaded!', bcolors.OKBLUE)
            
            #reads pair of images from one semantic class and and with binary labels
            if self.params['output_type'] == 'image_pair':
                items = self.db_items
                
                #In image_pair mode pair of images are sampled from the same semantic class
                clusters = PASCAL.cluster_items(self.db_items)
                
                #for set_id in clusters.keys():
                #    print clusters[set_id].length
                
                #db_items will be a list of tuples (set,j) in which set is the set that img_item belongs to and j is the index of img_item in that set
                self.db_items = []
                for item in items:
                    set_id = item.obj_ids[0]
                    imgset = clusters[set_id]
                    assert(imgset.length > self.params['k_shot']), 'class ' + imgset.name + ' has only ' + imgset.length + ' examples.'
                    in_set_index = imgset.image_items.index(item)
                    self.db_items.append((imgset, in_set_index))
                cprint('Total of ' + str(len(clusters)) + ' classes!', bcolors.OKBLUE)
        
        
        self.orig_db_items = copy.copy(self.db_items)

        assert(len(self.db_items) > 0), 'Did not load anything from the dataset'
        #assert(not self.params.has_key('db_cycle') or len(self.db_items) >= self.params['db_cycle']), 'DB Cycle should can not be more than items in the database = ' + str(len(self.db_items))
        #it forces the update_seq_index function to shuffle db_items and set seq_index = 0
        self.seq_index = len(self.db_items)
            

class PairLoaderProcess(Process):
    def __init__(self, name, queue, db_interface, params):
        Process.__init__(self, name=name)
        self.queue = queue
        self.db_interface = db_interface
        self.first_shape = params['first_shape']
        self.second_shape = params['second_shape']
        if params.has_key('shape_divisible'):
            self.shape_divisible = params['shape_divisible']
        else:
            self.shape_divisible = 1
        
        self.bgr = params['bgr']
        self.scale_256 = params['scale_256']
        self.first_label_mean = params['first_label_mean']
        self.first_label_scale = 1.0 if not params.has_key('first_label_scale') else params['first_label_scale']
        self.mean = np.array(params['mean']).reshape(1,1,3)
        self.first_label_params = params['first_label_params']
        self.second_label_params = params['second_label_params']
        self.deploy_mode = params['deploy_mode'] if params.has_key('deploy_mode') else False
        self.has_cont = params['has_cont'] if params.has_key('has_cont') else False
        if self.bgr:
            #Always store mean in RGB format
            self.mean = self.mean[:,:, ::-1]
            
    def run(self):
        try:
            while True:
                item = None
                while item is None:
                    item = self.load_next_frame()
                self.queue.put(item)
        except:
            cprint('An Error Happended in run()',bcolors.FAIL)
            cprint(str("".join(traceback.format_exception(*sys.exc_info()))), bcolors.FAIL)
            self.queue.put(None)
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    
    def load_next_frame(self, try_mode=True):
        next_pair = self.db_interface.next_pair()
        item = self.load_frame(*next_pair)
        # try to load the image more times in case it is None
        if item is None and not try_mode:
            item = self.try_some_more(100)
        return item

    # Tries to look for a valid image for a limited number of tries and then  
    # returns None if it doesn't find it
    def try_some_more(self, max_tries):
        i=0
        item =None
        while(item is None and i<max_tries):
            item = self.load_next_frame(True)
            i+=1
            print('Skipping image because of tiny object')
        return item

    def __prepross(self, frame_dict, shape = None):
        if frame_dict['mask'] is None: 
            return None
        
        image = frame_dict['image'] - self.mean
        label = frame_dict['mask']
         
        if shape is None:
            shape = np.array(image.shape[:-1], dtype=int)
        if self.shape_divisible != 1:
            shape = np.array(self.shape_divisible * np.ceil(shape / self.shape_divisible), dtype=np.int)     
        
        if tuple(shape) != image.shape[:-1]:
            image = resize(image, shape)
            label = resize(label, shape, order = 0, preserve_range=True)
            
        if self.bgr:
            image = image[:,:, ::-1]
            
        if self.scale_256:
            image *= 255
            
        return image, label, shape
    
    def __is_integer(self, mask):
      label_set = np.array(np.unique(mask), dtype=float)
      for label in label_set:
          if not label.is_integer():
              return False
      return True
    
    def __get_deploy_info(self, player, index):
        if index is None:
            return None, None, None
        if isinstance(player, ImagePlayer):
            img_item = player.image_item
            return img_item.obj_ids, img_item.read_mask(True), img_item.read_img()
        elif isinstance(player, VideoPlayer):
            img_item = player.video_item.image_items[index]
            return img_item.obj_ids, img_item.read_mask(True), img_item.read_img()
        else:
            raise Exception
    
    def load_frame(self, player, first_index, second_index):
        cprint('Loading pair = ' + player.name + ', ' + str(first_index) + ', ' + str(second_index), bcolors.WARNING)
        if second_index in first_index:
            return None
        
        
        images1 = []
        labels1 = []
        shape1 = self.first_shape
        for ind in first_index:
            frame1_dict = player.get_frame(ind)
            image1, label1, shape1 = self.__prepross(frame1_dict, shape1)
            images1.append(image1.transpose((2,0,1)))
            labels1.append(label1)
        item = dict(first_img=images1)
        
        if second_index is not None:
            frame2_dict = player.get_frame(second_index)
            image2, label2, shape = self.__prepross(frame2_dict, self.second_shape)
            item['second_img'] = [image2.transpose((2,0,1))]
        
        
        if self.deploy_mode:
            first_semantic_labels=[]
            first_mask_orig=[]
            first_img_orig=[]
            for ind in first_index:
                a,b,c = self.__get_deploy_info(player, ind)
                first_semantic_labels.append(a)
                first_mask_orig.append(b)
                first_img_orig.append(c)

            deploy_info = dict(seq_name=player.name, 
                               first_index=first_index, 
                               first_img_orig=first_img_orig,
                               first_mask_orig=first_mask_orig,
                               first_semantic_labels=first_semantic_labels)
            
            if second_index is not None:
                second_semantic_labels, second_mask_orig, second_img_orig = self.__get_deploy_info(player, second_index)
                deploy_info.update(second_index=second_index,
                                   second_img_orig=second_img_orig,
                                   second_mask_orig=second_mask_orig,
                                   second_semantic_labels=second_semantic_labels)
            
            item['deploy_info'] =  deploy_info
        
        #create first_labels
        for i in range(len(self.first_label_params)):
            name, down_scale, offset = self.first_label_params[i]
            item[name] = []
            for label1 in labels1:
                nlabel1 = change_coordinates(label1, down_scale, offset)
                nlabel1 = (nlabel1 - self.first_label_mean) * self.first_label_scale
                assert(self.__is_integer(nlabel1))
                item[name].append(nlabel1.reshape((1,) + nlabel1.shape))

        if second_index is not None:
            #create second_labels
            for i in range(len(self.second_label_params)):
                name, down_scale, offset = self.second_label_params[i]
                nlabel2 = change_coordinates(label2, down_scale, offset)
                assert(self.__is_integer(nlabel2))
                item[name] = [nlabel2.reshape((1,) + nlabel2.shape)]
        if self.has_cont:
            item['cont'] = [0] + [1] * (len(first_index) - 1)
            
        return item
            


class LoaderOfPairs(object):
    def __init__(self, profile):
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
	profile_copy['deploy_mode'] = True
        dbi = DBInterface(profile)
        self.PLP = PairLoaderProcess(None, None, dbi, profile_copy)
    def get_items(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (np.asarray(self.out['first_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['first_label']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_img']), #[np.newaxis,:,:,:], 
                np.asarray(self.out['second_label']), #[np.newaxis,:,:,:], 
                self.out['deploy_info'])
    def get_items_no_return(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
    # corrects the image to be displayable
    def correct_im(self, im):
        im = (np.transpose(im, (0,2,3,1)))/255.
        im += np.array([0.40787055,  0.45752459,  0.4810938])
        return im[:,:,:,::-1]
    # returns the outputs as images and also the first label in original img size
    def get_items_im(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (self.correct_im(self.out['first_img']), 
                self.out['original_first_label'], 
                self.correct_im(self.out['second_img']), 
                self.out['second_label'][0], 
                self.out['deploy_info'])


