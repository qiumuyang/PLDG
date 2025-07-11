# Partially Labeled Domain Generalization

```bash
bash start_par.sh <entrypoint> <domain_ids> <devices> <output_dir> <config> [config ...]

# This will run domain experiments on:
#   0,4 -> GPU0
#   1,5 -> GPU1
#   2   -> GPU2
#   3   -> GPU3
# Subdirectories for each domain will be created under <output_dir> like
#   runs/ours/A
#   runs/ours/B
#   ...
bash start_par.sh train_ours.py 0,1,2,3,4,5 0,1,2,3 \
    runs/ours/  \
    configs/dataset/abdomen.yaml \
    configs/classes/split1.yaml \
    configs/exp/base.yaml \
    configs/exp/bin.yaml \
    configs/exp/proto.yaml

# run for ablation or sotas, same as above
# remember to choose the correct config
bash start_par.sh ablation/<entrypoint> / sotas/<entrypoint>
```


```bash
bash eval.sh <domains> <devices> <dir> latest/best <target> <config> [<config> ...]

# target can be one / all / comma-separated-ids
bash eval.sh 0,1 0,1 runs/ours latest one configs/all_in_one.yaml

# for sotas, you should specify which sota by setting --sota=<name>
# refer to evaluate.py
bash eval.sh 0,1 0,1 runs/dodnet latest one configs/all_in_one.yaml --sota=dodnet
```
