import os
import xml.etree.ElementTree as ET

import numpy as np
from mmdet.apis import inference_detector
from mmdet.apis import init_detector

NAME_LABEL_MAP = {"huashang": 0, "wuzi": 1, "shakong": 2, "lianjiao": 3, "heidian": 4,
                  "yiwu": 5, "pifeng": 6, "guoshi": 7, "yanghua": 8, 'tongheidian': 9, 'danhuashang': 10}  # 类别字典


def get_all_boxes(txt_path, img_paths):
    with open(txt_path) as f:
        jpg = f.readlines()
        f.close()
    img_names = [c.strip() for c in jpg]
    all_boxes = []
    for name in img_names:
        img_path = os.path.join(img_paths, name + '.jpg')
        result = inference_detector(model, img_path)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        boxes = bboxes[:, :4].tolist()
        scores = bboxes[:, 4].tolist()
        classes = labels.tolist()
        img_box = []
        img_box.append(boxes)
        img_box.append(scores)
        img_box.append(classes)
        all_boxes.append(img_box)
    return all_boxes, img_names


# -----------写入检测结果-----------------------------------
def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
    for cls, cls_id in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        print("Writing {} VOC resutls file".format(cls))
        det_save_path = os.path.join(det_save_dir, "det_" + cls + ".txt")
        with open(det_save_path, 'wt') as f:
            for index, img_name in enumerate(test_imgid_list):
                this_img_detections = all_boxes[index]
                indexs = [index for index, class_id in enumerate(this_img_detections[2]) if class_id == cls_id]
                if len(indexs) == 0:
                    continue  # this cls has none detections in this img
                for i in indexs:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(img_name, this_img_detections[1][i],
                                   this_img_detections[0][i][0],
                                   this_img_detections[0][i][1],
                                   this_img_detections[0][i][2],
                                   this_img_detections[0][i][3]))  # that is [img_name, score, xmin, ymin, xmax, ymax]


# -----------------解析图片对应xml信息----------------
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    if os.path.exists(filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['ng_difficult'] = int(obj.find('ng_difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return objects
    else:
        return [{'name': None}]


# ------------计算ap-------------------------
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# --------------------voc评估--------------------------------------------
def voc_eval(detpath, annopath, test_imgid_list, cls_name, ovthresh=0.5,
             use_07_metric=False, use_diff=False):
    # 1. parse xml to get gtboxes
    # read list of images
    imagenames = test_imgid_list

    recs = {}
    for i, imagename in enumerate(imagenames):
        xml_path = os.path.join(annopath, imagename + '.xml')
        # if os.path.exists(xml_path):
        #     recs[imagename] = parse_rec(xml_path)
        # else:
        #     recs[imagename]=[]
        recs[imagename] = parse_rec(xml_path)
        # if i % 100 == 0:
        #   print('Reading annotation for {:d}/{:d}'.format(
        #     i + 1, len(imagenames)))

    # 2. get gtboxes for this class.
    class_recs = {}
    num_pos = 0
    # if cls_name == 'person':
    #   print ("aaa")
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
        bbox = np.array([x['bbox'] for x in R])
        if use_diff:
            difficult = np.array([False for x in R]).astype(np.bool)
        else:
            difficult = np.array([x['ng_difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        num_pos = num_pos + sum(~difficult)  # ignored the diffcult boxes
        class_recs[imagename] = {'bbox': bbox,
                                 'ng_difficult': difficult,
                                 'det': det}  # det means that gtboxes has already been detected

    # 3. read the detection file
    detfile = os.path.join(detpath, "det_" + cls_name + ".txt")
    with open(detfile, 'r') as f:
        lines = f.readlines()
    # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
    splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
    image_ids = [x[0] for x in splitlines]  # img_id is img_name
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    nd = len(image_ids)  # num of detections. That, a line is a det_box.
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]  # reorder the img_name

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]  # img_id is img_name
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['ng_difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # 4. get recall, precison and AP
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def do_python_eval(test_imgid_list, test_annotation_path, save_dir, flag, ovthresh):
    AP_list = []
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # color_list = colors.cnames.keys()[::6]

    for cls, index in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        recall, precision, AP = voc_eval(save_dir,
                                         test_imgid_list=test_imgid_list,
                                         cls_name=cls,
                                         annopath=test_annotation_path,
                                         use_07_metric=flag,
                                         ovthresh=ovthresh)
        AP_list += [AP]

        print("cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP))
    print("mAP is : {}".format(np.mean(AP_list)))


# ----------------voc检测评估-------------------------------------------------
def voc_evaluate_detections(all_boxes, test_annotation_path, test_imgid_list, save_dir, flag, ovthresh):
    '''
    :param all_boxes: is a list. each item reprensent the detections of a img.The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax].Note that: if none detections in this img. that the detetions is : []
    :return:
    '''
    # test_imgid_list = [item.split('.')[0] for item in test_imgid_list]
    write_voc_results_file(all_boxes, test_imgid_list=test_imgid_list,
                           det_save_dir=save_dir)
    do_python_eval(test_imgid_list, test_annotation_path=test_annotation_path, save_dir=save_dir, flag=flag,
                   ovthresh=ovthresh)


if __name__ == '__main__':
    voc_test_list = ['test.txt', 'target_text.txt']
    ovthresh_list = [0.1, 0.5]
    ZFM_list = ['ZM', 'FM']
    best_model = {'ZM': 'epoch_9.pth', 'FM': 'epoch_21.pth', }
    # 模型配置文件
    config_file = "/mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py"
    for ZFM in ZFM_list:
        # 预训练模型文件
        checkpoint_file = '/work_dir_' + ZFM + '/' + best_model[ZFM]
        # 通过模型配置文件与预训练文件构建模型
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        for ovthresh in ovthresh_list:
            for test_name in voc_test_list:
                print('-----{}-----{}-----{}'.format(ZFM, ovthresh, test_name.split('.')[0]))
                save_dir = './evaluate_output_' + test_name.split('.')[0] + '_' + str(ovthresh) + '_' + ZFM
                if os.path.exists(save_dir):
                    os.remove(save_dir)
                os.makedirs(save_dir)
                test_annotation_path = '/tmp/voc/' + ZFM + '/Annotations/'
                txt_path = '/tmp/voc/' + ZFM + '/' + test_name
                img_paths = '/tmp/coco_' + ZFM + '/' + test_name.split('.')[0] + 'set/'
                all_boxes, img_list = get_all_boxes(txt_path=txt_path, img_paths=img_paths)
                voc_evaluate_detections(all_boxes, test_annotation_path, img_list, save_dir, flag=False,
                                        ovthresh=ovthresh)
