import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
import numpy as np
import caffe, os, sys, cv2
import scipy
import pickle as pkl
import cPickle as cPkl



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
K = 10
BATCH = K+1
NMS_THRESH = 0.2
MODELS_DIR = '../models'
PARSE_DIR = '../parses/train'

def get_topK_boxes(scores, boxes, K):
    keep_boxes = []
    keep_scores = []
    keep_class = []
    for cls_ind, cls in enumerate(CLASSES):
        if cls_ind == 0:
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        keep_boxes.extend(cls_boxes[keep, :])
        keep_scores.extend(cls_scores[keep])
        keep_class.extend([cls_ind] * len(keep))
        
    mat_scores = np.array(keep_scores)
    mat_classes = np.array(keep_class)
    mat_boxes = np.array(keep_boxes, dtype='int16')       
    
    # get top K
    order = np.argsort(keep_scores)[::-1]
    keep = order[0:K]
    
    return (mat_boxes[keep, :], mat_scores[keep], mat_classes[keep])

def crop_from_boxes(img, boxes):
    cropped = []
    for idx in range(boxes.shape[0]):
        x1 = boxes[idx, 0]
        y1 = boxes[idx, 1]
        x2 = boxes[idx, 2]
        y2 = boxes[idx, 3]
        c = img[y1:y2, x1:x2, :]
        cropped.append(c)
    return cropped

def norm_box(boxes, img_width, img_height):
    norm_box = np.zeros((boxes.shape[0], boxes.shape[1]), dtype='float32')
    norm_box[:, [0,2]] = boxes[:, [0,2]] / float(img_width)
    norm_box[:, [1,3]] = boxes[:, [1,3]] / float(img_height)
    return norm_box

def gen_feature(img_array, net, box, cls_idx, score):
    batch_size = len(img_array)
    result = np.zeros((batch_size, 4122)) # 4096 + cls_idx + box + score 
    caffe_in = np.zeros((batch_size, 3,224,224))
    for i, img in enumerate(img_array):
        caffe_in[i, :, :, :] = transformer.preprocess('data', img)
        
    fc7 = net.forward_all(blobs=['fc7'], **{'data':caffe_in})
    result[:, 0:4096] = fc7['fc7']
    
    cls_flat_idx = [i*4122 + 4096 + v for i, v in enumerate(cls_idx)]
    result.flat[cls_flat_idx] = 1.0
    result[:, 4096+21:4096+25] = box # normalize by the height and width
    result[:, 4121] = score
    return result.ravel()


if __name__ == '__main__':  
    img_path = os.path.join(cfg.DATA_DIR, 'coco_train.txt')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    # load VGG 19
    vgg_deploy_path = MODELS_DIR + '/VGG_ILSVRC_19_layers_deploy.prototxt'
    vgg_model_path  = MODELS_DIR + '/VGG_ILSVRC_19_layers.caffemodel'
    net = caffe.Net(vgg_deploy_path, vgg_model_path, caffe.TEST)
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(MODELS_DIR + '/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    print "load VGG-19"

    with open(img_path, 'r') as f:
        img_list = f.readlines()
        img_list = [l.strip() for l in img_list]

    print 'read ' + str(len(img_list)) + ' images'


    parses_file = ['20000-parses.pkl', '30000-parses.pkl', '40000-parses.pkl', '50000-parses.pkl', '60000-parses.pkl', '70000-parses.pkl',
                   '80000-parses.pkl', 'final-parses.pkl']

    # init load
    print "loading first pkl"
    (scores, boxes) = pkl.load(open(os.path.join(PARSE_DIR, '10000-parses.pkl'), 'rb'))
    cnt = 0
    file_idx = 0
    portion_len = len(boxes)
    features = np.zeros((portion_len, 4122*BATCH))

    # read each images and get feature vectors
    for idx, img_file in enumerate(img_list):
        if cnt >= portion_len:
            print "loading new pkl and save results"
            (scores, boxes) = pkl.load(open(os.path.join(PARSE_DIR, parses_file[file_idx]), 'rb'))
            # save result
            resultSave = scipy.sparse.csr_matrix(features)
            resultSave32 = resultSave.astype('float32')
            cPkl.dump(resultSave32, open(os.path.join(PARSE_DIR, str(file_idx)+'-feature.pkl'), 'wb'))

            # clear faetures
            file_idx += 1
            portion_len = len(boxes)
            cnt = 0
            features = np.zeros((portion_len, 4122*BATCH))
   
        if idx % 1000==0:
            print "finish " + str(idx) + " for file " + str(file_idx)

        im_file = os.path.join(cfg.DATA_DIR, 'coco2014_train', img_file)
        raw_img = caffe.io.load_image(im_file)
        
        img_h = raw_img.shape[0]
        img_w = raw_img.shape[1]

        img_boxes = boxes[cnt]
        img_scores = scores[cnt]
        
        # get top K boxes
        (top_box, top_score, top_class) = get_topK_boxes(scores=img_scores, boxes=img_boxes,K=K)
        norm_top_box = norm_box(top_box, img_width=img_w, img_height=img_h)
              
        # crop images
        patches = crop_from_boxes(raw_img, top_box)

        # append whole image as 11th
        patches.append(raw_img)
        norm_top_box = np.vstack((norm_top_box, np.atleast_2d([0.0, 0.0, 1.0, 1.0])))
        top_class = np.append(top_class, 0)
        top_score = np.append(top_score, 1.0)
        
        x = gen_feature(patches, net, norm_top_box, top_class, top_score)
        features[cnt, :] = x    
        cnt += 1

    # save result
    resultSave = scipy.sparse.csr_matrix(features)
    resultSave32 = resultSave.astype('float32')
    cPkl.dump(resultSave32, open(os.path.join(PARSE_DIR, str(file_idx)+'-feature.pkl'), 'wb'))



















