# Uncertainty
This is the official implementation for 'Uncertainty Quantification in Stereo Matching'.

(Paper under review, arXiv version available soon).

## Install

Stereo and data uncertainty estimation:
```
pip install -r environment.yml
```

Model uncertainty estimation: please refer to [NUQ](https://github.com/stat-ml/NUQ).

## Datasets
Please refer to 'xxx_small.txt' in ./filenames folder for our splits.

We adopt [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
[KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo),
[Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/),
and [Driving Stereo](https://drivingstereo-dataset.github.io/) in stereo matching and uncertainty estimation.

## Run
Estimate stereo and data uncertainty on KITTI 2012+2015:
```
bash scripts/kitti1215.sh
```

## Citation

For any questions, please feel free to start an issue.

(Available soon).
