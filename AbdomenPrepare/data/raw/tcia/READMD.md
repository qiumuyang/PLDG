# TCIA

## Image

Please refer to:

https://www.cancerimagingarchive.net/collection/pancreas-ct/

Download requires [NBIA Data Retriever](https://www.cancerimagingarchive.net/ncia-download-tool/).
You can also use the [cli version](https://github.com/ygidtu/NBIA_data_retriever_CLI).

## Label

Please refer to the zenodo page for label data:

https://zenodo.org/records/1169361 *Multi-organ Abdominal CT Reference Standard Segmentations*

Extracted from `label_tciapancreasct_multiorgan.tar.gz` with `cropping.csv` ~~to get the bounding box for processing~~.

## Expected Folder Structure
```bash
data/raw/tcia/
├── image
│   ├── nbia_guest.json
│   ├── PANCREAS_0001
│   │   └── 11-24-2015
│   │       ├── 1.2.826.0.1.3680043.2.1125.1.68878959984837726447916707551399667.json
│   │       └── 1.2.826.0.1.3680043.2.1125.1.68878959984837726447916707551399667.zip
│   ├── PANCREAS_0002
│   │   └── 11-24-2015
│   │       ├── 1.2.826.0.1.3680043.2.1125.1.66376154825293205999744306285863502.json
│   │       └── 1.2.826.0.1.3680043.2.1125.1.66376154825293205999744306285863502.zip
│   ├── ...
├── label
│   ├── label0002.nii.gz
│   ├── label0003.nii.gz
│   ├── ...
└── READMD.md
```