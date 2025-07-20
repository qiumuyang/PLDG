import numpy as np

name_to_color = {
    "background": "#000000",
    "liver": "#FF191A",
    "spleen": "#00FF43",
    "left kidney": "#2F00CC",
    "right kidney": "#FAFF4A",
    "stomach": "#00FEFE",
    "gallbladder": "#E700DA",
    "esophagus": "#007777",
    "pancreas": "#FF90FF",
    "duodenum": "#CD8849",
    "colon": "#DFC199",
    "intestine": "#5DCEAC",
    "bladder": "#1F8C5B",
    "rectum": "#007676",
}
# src: https://zenodo.org/records/1169361
btcv_id_to_name = [
    "background",
    "spleen",
    "right kidney",  # btcv only
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",  # btcv only
    "inferior vena cava",  # btcv only
    "portal vein and splenic vein",  # btcv only
    "pancreas",
    "right adrenal gland",  # btcv only
    "left adrenal gland",  # btcv only
    "duodenum",
]
# WORD-V0.1.0: dataset.json
word_id_to_name = [
    "background",
    "liver",
    "spleen",
    "left kidney",
    "right kidney",
    "stomach",
    "gallbladder",
    "esophagus",
    "pancreas",
    "duodenum",
    "colon",
    "intestine",
    "adrenal",
    "rectum",
    "bladder",
    "Head_of_femur_L",
    "Head_of_femur_R",
]
# amos22: dataset.json
amos_id_to_name = [
    "background",
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
    "duodenum",
    "bladder",
    "prostate/uterus",
]


def parse_hex_to_tuple(color) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore


def make_palette(id_to_name, name_to_color):
    palette = []
    for name in id_to_name:
        if name not in name_to_color:
            palette.append("#000000")
        else:
            palette.append(name_to_color[name])
    return [parse_hex_to_tuple(color) for color in palette]


def partial_palette(palette, class_name, *names):
    # if not in names, set to background
    new_palette = []
    for i, color in enumerate(palette):
        name = class_name[i]
        if name in names:
            new_palette.append(color)
        else:
            new_palette.append((0, 0, 0))
    return new_palette


def partial_class(classes, *names):
    index = [0] * len(classes)
    for name in names:
        idx = classes.index(name)
        index[idx] = idx
    return np.array(index)


def make_class_mapping(id_to_name, shared_classes) -> list[int]:
    # 0 is reserved for background
    # return {
    #     i: shared_classes.index(name) if name in shared_classes else 0
    #     for i, name in enumerate(id_to_name)
    # }
    mp = [0] * len(id_to_name)
    for i, name in enumerate(id_to_name):
        if name in shared_classes:
            mp[i] = shared_classes.index(name)
    return mp


btcv_palette = make_palette(btcv_id_to_name, name_to_color)
# tcia_palette = btcv_palette
word_palette = make_palette(word_id_to_name, name_to_color)
amos_palette = make_palette(amos_id_to_name, name_to_color)

# [
#     'background', 'spleen', 'right kidney', 'left kidney', 'gallbladder',
#     'esophagus', 'liver', 'stomach', 'pancreas', 'duodenum'
# ]
# 0: background
# 1: spleen
# 2: left kidney
# 3: right kidney
# 4: gallbladder
# 5: esophagus
# 6: liver
# 7: stomach
# 8: pancreas
# 9: duodenum
shared_classes = set.intersection(set(btcv_id_to_name), set(word_id_to_name),
                                  set(amos_id_to_name))
shared_classes = sorted(shared_classes, key=lambda x: btcv_id_to_name.index(x))
shared_palette = make_palette(shared_classes, name_to_color)

# map from original class name to new index
btcv_class_mapping = make_class_mapping(btcv_id_to_name, shared_classes)
# tcia_class_mapping = btcv_class_mapping
word_class_mapping = make_class_mapping(word_id_to_name, shared_classes)
amos_class_mapping = make_class_mapping(amos_id_to_name, shared_classes)
