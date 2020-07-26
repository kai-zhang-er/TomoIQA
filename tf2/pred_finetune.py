import tensorflow as tf
from src.net.model_v2 import VGGIQAModel
import cv2


finetune_model_path = "experiments/TOMO/rankiqa2_finetune/model.h5"
txt_file_path="/content/drive/My Drive/pyIQA/TOMO/test_finetune.txt"
data_dir="/content/drive/My Drive/pyIQA/dataset/testset/"
result_file_path="./predict_result.txt"

def test():
    model = VGGIQAModel(is_training=False)
    model.summary()
    model.load_weights(finetune_model_path)
    f_result=open(result_file_path,"w")
    f_test=open(txt_file_path,"r")
    lines=f_test.readlines()
    lines.sort()
    for line in lines:
        items=line.strip("\n").split(",") # image_name, gt_score
        image=cv2.imread(data_dir+items[0])
        score=model.predict(image)
        print("{}->{}".format(items[0],score))

    tf.print("finish")
    f_result.close()
    f_test.close()

if __name__=="__main__":
    test()