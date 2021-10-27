import numpy as np
import os, json, cv2

bdd_dir = "/home/kazuki.konishi.rn/ObjectDetection/bdd100k/"
out_dir = "/home/kazuki.konishi.rn/ObjectDetection/Causal-based-OD/data/"
train_or_val = "train"

weather = "rainy"
attributes = {"weather": weather}
# - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
# - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
# - timeofday: "daytime|night|dawn/dusk|undefined"

img_dir = bdd_dir +"images/100k/" + train_or_val
input_file = bdd_dir + "labels/bdd100k_labels_images_" + train_or_val + ".json"
output_file = out_dir + "bdd100k_" + weather + "_" + train_or_val + ".json"

category_dict = {
    1: "person",
    2: "rider",
    3: "car",
    4: "bus",
    5: "truck",
    6: "bike",
    7: "motor",
    8: "traffic light",
    9: "traffic sign",
    10: "train",
}

def set_bdd100k_category(category_dict):
    categories = []
    for id, name in category_dict.items():
        category = {
            "supercategory": "none",
            "id": id, 
            "name": name
        } 
        categories.append(category)
    return categories


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


def get_bdd100k_dicts(input_file, img_dir, **kwargs):
    json_file = os.path.join(input_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    images = []
    annotations = []
    idx = 1
    for v in imgs_anns:
        flag = True
        if len(kwargs) != 0:
            for key, attribute in kwargs.items():
                flag = flag and (v["attributes"][key] == attribute)
        if flag:
            image = {}
            try:
                filename = os.path.join(img_dir, v["name"])
                height, width = cv2.imread(filename).shape[:2]
            except:
                continue
            image["file_name"] = v["name"]
            image["height"] = height
            image["width"] = width
            image["id"] = idx
            images.append(image)
            idx += 1

            annos = v["labels"]
            for anno in annos:
                try:
                    box = anno["box2d"]
                except:
                    continue
                width = box["x2"]-box["x1"]
                height = box["y2"]-box["y1"]
                obj = {
                    "iscrowd" : 0,
                    "image_id": image["id"],
                    "bbox": [box["x1"], box["y1"], width, height],
                    "area" : width*height,
                    "category_id": get_key_from_value(category_dict, anno["category"]),
                    "ignore" : 0,
                    "id" : anno["id"]
                }
                annotations.append(obj)

    return images, annotations


if __name__ == "__main__":
    categories = set_bdd100k_category(category_dict)
    images, annotations = get_bdd100k_dicts(input_file, img_dir, **attributes)
    out_dict = {
        "categories" : categories,
        "images" : images,
        "annotations" : annotations,
        "type" : "instances"
    }
    with open(output_file, 'w') as f:
        json.dump(out_dict, f, indent=None)
