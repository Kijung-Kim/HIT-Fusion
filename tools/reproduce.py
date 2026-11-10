"""CPU reproduction driver. Never derives ground truth from predictions or names."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover(root):
    result = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('_'):
            continue
        # The original installation retained several camera streams. Only 506cbd
        # is used by the published driver; flattened folders are accepted too.
        candidates = [folder / '506cbd', folder]
        if (folder / '506cbd' / 'a').is_dir():
            candidates.insert(0, folder / '506cbd' / 'a')
        selected = next((p for p in candidates if list(p.glob('*.jpg'))), None)
        if selected:
            result.append((folder.name, selected))
        else:
            raise ValueError(f'No unambiguous frame directory for {folder}')
    return result


def evaluate(gt, predictions):
    if not isinstance(gt, dict) or not gt:
        raise ValueError('GT must be a nonempty object keyed by sequence ID')
    if set(gt) != set(predictions):
        raise ValueError('GT and predictions must contain exactly the same sequence IDs')
    tp = fp = fn = positive = 0
    verified_tn = verified_negative = mixed = mixed_no_output = 0
    details = {}
    for name, reference in gt.items():
        predicted = predictions[name]['prediction_frame_id']
        if predictions[name]['status'] == 'error':
            raise ValueError(f'Execution error in {name}; fix it before scoring')
        kind = reference['kind']
        if kind == 'impact':
            targets = reference['frames']
            if not targets or any(type(t) is not int or t < 0 for t in targets):
                raise ValueError(f'Invalid annotated frame set: {name}')
            positive += 1
            if predicted in targets:
                tp += 1
                details[name] = 'matched'
            elif predicted >= 0:
                fp += 1
                fn += 1
                details[name] = 'wrong_frame'
            else:
                fn += 1
                details[name] = 'missing_output'
        elif kind == 'verified_nonimpact':
            verified_negative += 1
            verified_tn += predicted < 0
            details[name] = 'verified_tn' if predicted < 0 else 'verified_fp'
        elif kind == 'mixed_or_indeterminate':
            mixed += 1
            mixed_no_output += predicted < 0
            details[name] = 'unscored_mixed'
        else:
            raise ValueError(f'Unknown reference kind: {kind}')
    # These metrics deliberately match Table III's impact-only population.
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive if positive else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return dict(annotated_impacts=positive, tp=tp, fp=fp, fn=fn,
                localization_success=recall, precision=precision, recall=recall, f1=f1,
                verified_nonimpact=verified_negative, verified_tn=verified_tn,
                verified_fp=verified_negative-verified_tn,
                mixed=mixed, mixed_no_output=mixed_no_output, details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gt', type=Path)
    parser.add_argument('--profile', choices=['v3','v4','v4_ball','v4_temporal','v4_occlusion'], default='v4')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    repo = Path(__file__).resolve().parents[1]
    exe = repo / 'x64/Release/hit_paper.exe'
    if not exe.exists():
        raise FileNotFoundError('Run build_paper.ps1 first')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences = discover(args.data.resolve())
    if args.shards<1 or not 0<=args.shard_index<args.shards:
        raise ValueError('Invalid shard selection')
    sequences = sequences[args.shard_index::args.shards]
    if args.limit is not None:
        sequences = sequences[:args.limit]
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))  # Official yolov8n-seg.pt, no fine-tuning.
    torch.set_num_threads(1)
    predictions = {}
    metadata = dict(model=str(args.model.resolve()), model_sha256=sha256(args.model),
                    torch=torch.__version__, ultralytics=ultralytics.__version__,
                    segmentation_opencv=cv2.__version__, segmentation_input=640,
                    segmentation_confidence=0.25, mask_box_matching='maximum IoU > 0',
                    detector_sha256=sha256(repo / 'yolo.weights'),
                    exe_sha256=sha256(exe),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()),
                    original_segmentation_checkpoint_identity='unavailable',
                    pipeline_threads=1, frame_id_convention='numeric filename stem',
                    ground_truth=str(args.gt.resolve()) if args.gt else None,
                    pipeline_profile=args.profile)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for name, folder in sequences:
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        files = sorted(folder.glob('*.jpg'), key=lambda f: int(f.stem))
        start = time.perf_counter()
        result = model.predict(str(files[min(10,len(files)-1)]), imgsz=640,
                               classes=[0], conf=0.25, device='cpu',
                               retina_masks=True, verbose=False)[0]
        segment_seconds = time.perf_counter()-start
        fs = cv2.FileStorage(str(directory/'segmentation.yml'),cv2.FILE_STORAGE_WRITE)
        fs.startWriteStruct('instances',cv2.FileNode_SEQ)
        if result.masks is not None:
            for index, (box,mask) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.masks.data.cpu().numpy())):
                mask_path = directory / f'person_{index}.png'
                cv2.imwrite(str(mask_path),(mask>0.5).astype(np.uint8)*255)
                x,y,x2,y2 = [int(v) for v in box]
                fs.startWriteStruct('',cv2.FileNode_MAP)
                for key,value in dict(x=x,y=y,w=x2-x,h=y2-y,mask=str(mask_path).replace('\\','/')).items():
                    fs.write(key,value)
                fs.endWriteStruct()
        fs.endWriteStruct()
        fs.release()
        env = os.environ.copy()
        opencv = Path(env.get('OpenCV_DIR','C:/opencv/opencv340/build'))/'x64/vc15/bin'
        env['PATH'] = str(opencv) + os.pathsep + env.get('PATH','')
        command=[str(exe),str(folder),str(directory/'segmentation.yml'),
                 str(directory/'result.yml'),str(directory/'signals.csv'),args.profile]
        with (directory/'execution.log').open('w',encoding='utf-8') as log:
            process = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        output = cv2.FileStorage(str(directory/'result.yml'),cv2.FILE_STORAGE_READ)
        if not output.isOpened():
            raise RuntimeError(f'No result for {name}, return code {process.returncode}; see execution.log')
        entry = {k:output.getNode(k).string() for k in ['status','error']}
        entry.update({k:int(output.getNode(k).real()) for k in ['prediction_frame_id','candidate_record_index','verified_anchor_record_index','ball_x','ball_y','frames']})
        entry['pipeline_seconds']=output.getNode('pipeline_seconds').real()
        entry['segmentation_seconds']=segment_seconds
        entry['sequence_seconds']=time.perf_counter()-start
        entry['return_code']=process.returncode
        output.release()
        predictions[name]=entry
        (args.output/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
        print(name,entry['status'],entry['prediction_frame_id'],round(entry['sequence_seconds'],3),flush=True)
    if args.gt:
        gt = json.loads(args.gt.read_text(encoding='utf-8-sig'))
        metrics = evaluate(gt,predictions)
        (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(json.dumps(metrics,indent=2))
    else:
        print('Predictions saved. Accuracy is unverified: no ground truth supplied.')


if __name__ == '__main__':
    main()
