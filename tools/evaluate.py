"""Score retained predictions without re-running image processing."""
import argparse
import json
from pathlib import Path
from reproduce import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    predictions=json.loads(args.predictions.read_text(encoding='utf-8-sig'))
    gt=json.loads(args.gt.read_text(encoding='utf-8-sig'))
    metrics=evaluate(gt,predictions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
