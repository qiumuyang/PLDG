# BTCV

## Image

Please refer to the Synapse page for image data:

https://www.synapse.org/Synapse:syn3193805/wiki/89480 *Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge*

Extracted from `Abdomen/RawData.zip`: RawData/Training/img and RawData/Testing/img

## Label

Please refer to the zenodo page for label data:

https://zenodo.org/records/1169361 *Multi-organ Abdominal CT Reference Standard Segmentations*

Extracted from `label_btcv_multiorgan.tar.gz` with `cropping.csv` ~~to get the bounding box for processing~~.

## Expected Folder Structure

```bash
data/raw/btcv/
├── image
│   ├── img0001.nii.gz
│   ├── img0002.nii.gz
│   ├── ...
├── label
│   ├── label0001.nii.gz
│   ├── label0002.nii.gz
│   ├── ...
└── README.md
```