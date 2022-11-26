import os
import json
from keras.models import load_model
import pickle
import numpy as np
import cv2              
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import io


THU_MUC_GOC = r"F:\Code\DistractedDriverDetection_Final\demo\demo_Image"
# BASE_DIR = "D:\Code\Project\DistractedDriverDetection"
# PICKLE_DIR = os.path.join(THU_MUC_GOC,"pickle_files")
BEST_MODEL = os.path.join(THU_MUC_GOC,"distracted-22-0.98.hdf5")
model = load_model(BEST_MODEL)

# with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
#     labels_id = pickle.load(handle)

labels_id = {'c3': 0, 'c9': 1, 'c1': 2, 'c6': 3, 'c2': 4, 'c7': 5, 'c8': 6, 'c4': 7, 'c5': 8, 'c0': 9}
def path_to_tensor(img_path):
    img = load_img(img_path, target_size=(128, 128))
    x = img_to_array(img)
    # img = cv2.imread(img_path)
    # x = Image.fromarray(img)
    return np.expand_dims(x, axis=0)

# def paths_to_tensor(img_paths):
    # list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    # return np.vstack(list_of_tensors)

def return_prediction(filename):
    buffer = io.BytesIO()
    filename.save(buffer, 'jpeg')
    buffer.seek(0)
    filename = buffer
    test_tensors = path_to_tensor(filename).astype('float32')/255 - 0.5

    ypred_test = model.predict(test_tensors,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    print('label',labels_id)
    print(ypred_class)
    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    # print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]
    # print('res',res)

    with open(os.path.join(os.getcwd(),'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
    prediction_result = info[res]
    return prediction_result

if __name__=='__main__':
    pass
