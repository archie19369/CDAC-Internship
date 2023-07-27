import torch, torchvision, detectron2

# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

# COMMON LIBRARIES
import os
import cv2

from datetime import datetime
import cv2

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

# from roboflow import Roboflow

# rf = Roboflow(api_key="TRFZ41J5IvdEFvupo6SQ")
# project = rf.workspace("cdac-0ls9l").project("maize-quality-detection-5klwo")
# dataset = project.version(1).download("coco")

import multiprocessing



DATA_SET_NAME = 'Maize-Quality-Detection'
DATA_SET_LOCATION = 'C:\\Users\\archa\\Desktop\\C-DAC\\Using_Detectron2(6 classes)\\Maize_Detection_Using_Detectron2\\Maize-Quality-Detection-1'
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_LOCATION, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_LOCATION, "train", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME, 
    metadata={}, 
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH, 
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_LOCATION, "test")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_LOCATION, "test", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TEST_DATA_SET_NAME, 
    metadata={}, 
    json_file=TEST_DATA_SET_ANN_FILE_PATH, 
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_LOCATION, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_LOCATION, "valid", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=VALID_DATA_SET_NAME, 
    metadata={}, 
    json_file=VALID_DATA_SET_ANN_FILE_PATH, 
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)

print([
    data_set
    for data_set
    in MetadataCatalog.list()
    if data_set.startswith(DATA_SET_NAME)
])

import cv2
import matplotlib.pyplot as plt




os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
    multiprocessing.freeze_support()

   
    
    metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
    

    dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

    dataset_entry = dataset_train[0]
    image = cv2.imread(dataset_entry["file_name"])

    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata=metadata, 
        scale=0.8, 
        instance_mode=ColorMode.IMAGE_BW
    )

    out = visualizer.draw_dataset_dict(dataset_entry)

# plt.imshow(out.get_image()[:, :, ::-1])
# plt.axis('off')  # Optional: To hide the axes
# plt.show()

# HYPERPARAMETERS
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 1000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 6

# OUTPUT DIR
OUTPUT_DIR_PATH = os.path.join(
    DATA_SET_NAME, 
    ARCHITECTURE, 
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
)


os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = ()
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH

print("Entering into training")
if __name__ == "__main__":
    multiprocessing.freeze_support()
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


# Predicting on unknown images
if __name__ == "__main__":
    multiprocessing.freeze_support()
    predictor = DefaultPredictor(cfg)

    # Path to the directory containing images
    image_dir = "C:\\Users\\archa\Desktop\\C-DAC\\Using_Detectron2(6 classes)\\Maize_Detection_Using_Detectron2\\Maize-Quality-Detection-1\\test"

    # List all image file names in the directory
    image_files = os.listdir(image_dir)

    # Loop over each image file
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        # Make prediction
        outputs = predictor(image)

        # Visualize and display the prediction
        visualizer = Visualizer(
            image[:, :, ::-1],
            metadata=metadata, 
            scale=0.8, 
            instance_mode=ColorMode.IMAGE_BW
        )
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Output directory to save the images with predictions
        output_dir = "C:\\Users\\archa\\Desktop\\C-DAC\\Using_Detectron2(6 classes)\\Maize_Detection_Using_Detectron2\\Maize-Quality-Detection-1\\prediction"

        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])

        # plt.imshow(out.get_image()[:, :, ::-1])
        # plt.axis('off')  # Optional: To hide the axes
        # plt.show()