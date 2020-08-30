import tensorflow as tf
from src.net.model_v2 import EfficientIQAModel
import cv2
from utils import ex


@ex.main
def test():
    # predict quality scores
    finetune_model_path = save_ckpt_dir+finetune_model_name+"/model.h5"
    txt_file_path=config_dir+"test_finetune.txt"

    model = EfficientIQAModel(is_training=False)
    model.summary()
    model.load_weights(finetune_model_path)

    # read the image name
    f_test=open(txt_file_path,"r")
    lines=f_test.readlines()
    f_test.close()

    lines.sort()
    for line in lines:
        items=line.strip("\n").split(",") # image_name, gt_score
        image=cv2.imread(testdir+items[0])
        score=model.predict(image)
        print("{}->{}".format(items[0],score))

    tf.print("finish")

