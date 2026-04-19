import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    # ── Parse options ──────────────────────────────────────────────────────────
    source   = opt.source
    weights  = opt.weights
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz    = opt.img_size
    trace    = not opt.no_trace

    save_img = not opt.nosave and not source.endswith('.txt')
    webcam   = (
        source.isnumeric()
        or source.endswith('.txt')
        or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    )

    # ── Output directories ─────────────────────────────────────────────────────
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # ── Device setup ───────────────────────────────────────────────────────────
    set_logging()
    device = select_device(opt.device)
    half   = device.type != 'cpu'  # Half precision only supported on CUDA

    # ── Load model ─────────────────────────────────────────────────────────────
    model  = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz  = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    # ── Optional second-stage classifier ───────────────────────────────────────
    # Set classify=True to enable a ResNet101 second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt', map_location=device)['model']
        ).to(device).eval()

    # ── Dataloader ─────────────────────────────────────────────────────────────
    vid_path, vid_writer = None, None

    if webcam:
        view_img        = check_imshow()
        cudnn.benchmark = True
        dataset         = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ── Class names and bounding-box colors ────────────────────────────────────
    names  = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # ── Warm-up run ────────────────────────────────────────────────────────────
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    k  = 0  # Crop counter

    # ── Inference loop ─────────────────────────────────────────────────────────
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Re-warm if image shape changed (webcam resolution switch, etc.)
        if device.type != 'cpu' and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=opt.augment)[0]

        # Forward pass
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Non-Maximum Suppression
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # ── Process detections ─────────────────────────────────────────────────
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p         = Path(p)
            save_path = str(save_dir / p.name)
            txt_path  = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}'
            )
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n  = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    # Save label to txt
                    if save_txt:
                        xywh = (
                            xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
                        ).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Crop detected region with 10% margin on each side
                    xyxy     = [int(x) for x in xyxy]
                    width    = xyxy[2] - xyxy[0]
                    height   = xyxy[3] - xyxy[1]
                    margin_x = int(width  * 0.1)
                    margin_y = int(height * 0.1)

                    x1 = max(0,            xyxy[0] - margin_x)
                    y1 = max(0,            xyxy[1] - margin_y)
                    x2 = min(im0.shape[1], xyxy[2] + margin_x)
                    y2 = min(im0.shape[0], xyxy[3] + margin_y)

                    cropped_img    = im0[y1:y2, x1:x2]
                    crop_dir       = save_dir / 'crops'
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_save_path = crop_dir / f'{p.stem}_crop_{k}.jpg'
                    cv2.imwrite(str(crop_save_path), cropped_img)
                    k += 1

            print(
                f'{s}Done. '
                f'({1E3 * (t2 - t1):.1f}ms) Inference, '
                f'({1E3 * (t3 - t2):.1f}ms) NMS'
            )

            # Display frame
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save output
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f"Result saved to: {save_path}")
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)         if vid_cap else 10
                        w   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if vid_cap else im0.shape[1]
                        h   = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
                        save_path  += '.mp4'
                        vid_writer  = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        n_labels = len(list(save_dir.glob('labels/*.txt')))
        s = f"\n{n_labels} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f'Done. ({time.time() - t0:.3f}s){s}')


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',      nargs='+', type=str,   default='yolov7.pt',        help='model.pt path(s)')
    parser.add_argument('--source',       type=str,              default='inference/images',  help='file/dir/URL/glob/0(webcam)')
    parser.add_argument('--img-size',     type=int,              default=640,                 help='inference size in pixels')
    parser.add_argument('--conf-thres',   type=float,            default=0.25,                help='object confidence threshold')
    parser.add_argument('--iou-thres',    type=float,            default=0.45,                help='IOU threshold for NMS')
    parser.add_argument('--device',                              default='',                  help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--view-img',     action='store_true',                                help='display results')
    parser.add_argument('--save-txt',     action='store_true',                                help='save results to *.txt')
    parser.add_argument('--save-conf',    action='store_true',                                help='save confidences in --save-txt labels')
    parser.add_argument('--nosave',       action='store_true',                                help='do not save images/videos')
    parser.add_argument('--classes',      nargs='+', type=int,                                help='filter by class index')
    parser.add_argument('--agnostic-nms', action='store_true',                                help='class-agnostic NMS')
    parser.add_argument('--augment',      action='store_true',                                help='augmented inference')
    parser.add_argument('--update',       action='store_true',                                help='update all models')
    parser.add_argument('--project',                             default='runs/detect',       help='save results to project/name')
    parser.add_argument('--name',         type=str,              default='exp',               help='save results to project/name')
    parser.add_argument('--exist-ok',     action='store_true',                                help='do not increment run folder')
    parser.add_argument('--no-trace',     action='store_true',                                help="don't trace model")

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
