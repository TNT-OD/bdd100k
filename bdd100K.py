import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer


category_dict = {
    0: "person", #not pedestrian??
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motor", #motorcycle,
    7: "bike",# bicycle
    8: "traffic light",
    9: "traffic sign",
}

things_list = [category_dict[i] for i in range(10)]

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def get_bdd100k_dicts(img_dir, trainorval):
    json_file = os.path.join(img_dir, "labels", "bdd100k_labels_images_"+trainorval+".json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        try:
            filename = os.path.join(img_dir, "images", "100k", trainorval, v["name"])
            height, width = cv2.imread(filename).shape[:2]
        except:
            continue
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["labels"]
        objs = []
        for anno in annos:
            try:
                box = anno["box2d"]
            except:
                continue
            obj = {
                "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": get_key_from_value(category_dict, anno["category"]),
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



if __name__ == "__main__":
    for d in ["train", "val"]:
        DatasetCatalog.register("bdd100K_" + d, lambda d=d: get_bdd100k_dicts( "/home/konishi/detectron2/bdd100k", d))
        MetadataCatalog.get("bdd100K_" + d).set(thing_classes=things_list)
    bdd100K_metadata = MetadataCatalog.get("bdd100K_train")
    dataset_dicts = get_bdd100k_dicts("/home/konishi/detectron2/bdd100k", "train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("bdd100K_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    # train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_100K.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_bdd100k_dicts("/home/konishi/detectron2/bdd100k", "val")
    fig = plt.figure(figsize=(16, 16))

    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=bdd100K_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(out.get_image()[:, :, :])
        ax.axis("off")