from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from utils.dataloaders import LoadImages,LoadStreams
from utils.torch_utils import select_device, time_sync
from ultralytics.utils.plotting import Annotator, colors
from upload_data import upload_data



import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json
import datetime

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class_dict = {}
dir_data = {}
top_id = []
save_data = False
miss_count = 0


def detect(opt):
    global save_data
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    for i, name in enumerate(names):  
        if name not in class_dict:
            class_dict[i] = 0

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        # print(f"Image: {img.shape} ")
        # print(f"Image Type: {type(img)} ")
        
        t1 = time_sync()


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1


        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
       
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            start_time = time.time()
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            # print(f"W: {w} h {h}")F
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                #print(f"Outputs: {outputs}")

                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    # If save flag, save image and bounding boxes for uploading
                    marker_object_tracking(outputs)
                    if save_data:
                            save(im0s,str(outputs))
                            save_data = False
                            
                            
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # print(f"Img: {im0.shape}\n")
                        moving =  motion(id,bboxes[1])

                        if moving:
                            count_obj(bboxes,w,h,id,int(cls),class_dict)
                        # print(im0.shape)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                #LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
                    

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
                            
            if show_vid:
                
                color=(0,0,255)
                # print(f"Shape: {im0.shape}")

                # draw lines
                cv2.line(im0, (0, int(h/2)), (w,int(h/2)), (255,0,0), thickness=3)
                cv2.line(im0, (0, int(h/2)-300), (w,int(h/2)-300), (255,0,0), thickness=3)

                
                
                thickness = 3 # font thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.2 
                cv2.putText(im0, "Outgoing Trash:  "+str(sum(class_dict.keys())), (60, 150), font, 
                   fontScale, (99,185,245), thickness, cv2.LINE_AA)

        
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(im0, "FPS: " + str(int(fps)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                
                try :
                    cv2.imshow('Baltimore_AI', im0)
                    if cv2.waitKey(1) % 256 == 27:  # ESC code 
                        print('Totals: ', class_dict)
                        vid_writer.write(im0)
                        raise StopIteration  
                except KeyboardInterrupt:
                    raise StopIteration
                

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        #print('Results saved to %s' % save_path)
        os.system('open ' + save_path)
        
    print(f"Totals: {class_dict}")



def count_obj(box,w,h,id,cls,class_dict):

  
    #find center of the box 
    cx, cy = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    # if the object isn't in the defined tracking area, ignore it
    if cy>= int(h//2) or cy<= int(h//2)-300:
        return
            
    if cy < (int(h/2)-20):
                
        if cls not in class_dict:
            class_dict[cls] = 1
        else:
            class_dict[cls] += 1  

def motion(id,y):
    ''' determines if objects are moving'''
    global dir_data

    dir_data[id] = y
    #sys.stdout.write(f'\r {dir_data}')

    if dir_data[id] - y < 0:
        
        return False
    else:
        
        return True
    
def marker_object_tracking(outputs):
    global dir_data
    global top_id
    global save_data
    global miss_count
    
    for output in outputs:
        id = output[4]
        y = output[1]

        if id not in dir_data:
            dir_data[id] = y

    for key in list(dir_data.keys()):
        if key not in outputs[:,4]:
            del dir_data[key]
            
    if len(top_id) > 0:
        if top_id[0] not in dir_data:
            miss_count += 1 
            if miss_count > 90:
                print("Missed object for too long")
                top_id.clear()
                get_top_id()
                save_data = False
                miss_count = 0
        else:
            miss_count = 0
            if dir_data[top_id[0]] < 200:
                save_data = True
                top_id.clear()
                get_top_id()
            else: 
                save_data = False

    elif len(top_id) == 0:
        get_top_id()
        

    #sys.stdout.write(f'\r {dir_data}')
    #print(f"Dir Data: {dir_data}")
    #print(f'{len(dir_data)} objects being tracked')

def get_top_id():

    global top_id
    global dir_data
    id = max(dir_data, key=dir_data.get)
    if dir_data[id] < 1270:
        top_id.clear()
        return
    else:
        top_id.append(id)
        top_id.append(dir_data[top_id[0]])

    print(f"Top ID: {top_id}")

def save(img,annotations):
    global class_dict
    ''' Save images and annotations to a folder '''
    print("Saving data")
    name = str(int(time.time()))
    image_save_dir = Path('inference/output')
    annotations_save_dir = Path('inference/output/annotations')

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    if not os.path.exists(annotations_save_dir):
        os.makedirs(annotations_save_dir)

    file = (f"{image_save_dir/name}.jpg")
    cv2.imwrite(file,img)
    with open((annotations_save_dir/name).with_suffix('.txt'), 'a') as file:
        file.write(annotations)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    timestamp = now.strftime('%m-%d-%y %H:%M:%S')
    data = class_dict.copy()
    data['timestamp'] = timestamp
    data['totals'] = sum(class_dict.values())
    json.dumps(data)
    upload_data(device_id=2, image_file_path=f"{image_save_dir}/{name}.jpg", bounding_box_file_path=f"{annotations_save_dir}/{name}.txt",data=data)
    print("Data saved")
    # os.remove(f"{image_save_dir/name}.jpg")
    # os.remove(f"{annotations_save_dir/name}.txt")
    return
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='input.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='MJPG', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default='store_true', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)

