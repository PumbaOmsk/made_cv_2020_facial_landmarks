import os
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm.notebook as tqdm
from torch.utils import data

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 128
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"

def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):        
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)
               
        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]
        coef = batch["scale_coef"].numpy()
        coef = 1/coef
        coef = coef*coef
        coef =torch.tensor(coef)
        loss_fn.setWeight(coef)
        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks)
        val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        if "dx" in batch.keys():
            dx = batch["dx"].numpy()  # B
            dy = batch["dy"].numpy()  # B
            prediction = restore_landmarks_batch_ex(pred_landmarks, fs, margins_x, margins_y,dx,dy)  # B x NUM_PTS x 2
        else:
            prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions

class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape        
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample

class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample

class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train", size=None, ignore_image=None): # , isFlip=False
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []
#         self.flip_landmarks = []

        if size is None:            
            with open(landmark_file_name, "rt") as fp:
                num_lines = sum(1 for line in fp)
        else:
            num_lines = size-1

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp)):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reached start of val part of data
                if i>=int(num_lines):
                    break # чтобы можно было грузить меньше картинок
                elements = line.strip().split("\t")
                if ignore_image is not None and elements[0] in ignore_image:
                    print("ignore: ",elements[0])
                    continue
                    
                image_name = os.path.join(images_root, elements[0])                
                self.image_names.append(image_name)            
                
                if split in ("train", "val"):
                    landmarks = list(map(np.int16, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int16).reshape((len(landmarks) // 2, 2))                                                            
                    self.landmarks.append(landmarks)
#                     flip_landmarks = flip_lm_v(landmarks)
#                     self.flip_landmarks.append(flip_landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
            # self.flip_landmarks = torch.as_tensor(self.flip_landmarks)
        else:
            self.landmarks = None
            # self.flip_landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                
        sample["image"] = image
        
        if self.landmarks is not None:
            landmarks = self.landmarks[idx].clone()
#             flip_landmarks = self.flip_landmarks[idx].clone()
            
            sample["landmarks"] = landmarks
#             sample["flip_landmarks"] = flip_landmarks
        
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks_batch_ex(landmarks, fs, margins_x, margins_y,dx,dy):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    landmarks[:, :, 0] += dx[:, None]
    landmarks[:, :, 1] += dy[:, None]
    return landmarks
    
def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]    
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')

def draw_landmarks(image, landmarks):
    for point in landmarks:
        x, y = point.astype(np.int)
        cv2.circle(image, (x, y), 1, (128, 0, 128), 1, -1)
    return image

# функции отражения разметки. В итоговой модели не пригодились
# def flip_lm_width(lm,shape):
#     lmtr = lm.copy()    
#     height, width, _ = shape        
#     lmtr[:,0] = width-lmtr[:,0]    
        
#     lmf = np.zeros(len(lm)*2).reshape(-1,2);            
#     lmf[0:64] = lmtr[64:128]  # овал лица лево низ
#     lmf[64:128]= lmtr[0:64]   # овал лица право низ
#     lmf[128:200]= lmtr[272:200:-1] # овал лица право верх
#     lmf[200] = lmtr[200] # центр на лбу
#     lmf[201:273]= lmtr[199:127:-1] # овал лица лево верх
#     lmf[273:273+64]= lmtr[337:337+64] # левая бровь 
#     lmf[337:337+64]= lmtr[273:273+64] # правая бровь    
#     lmf[401:401+63]= lmtr[464:464+63] # левая душка носа
#     lmf[464:464+63]= lmtr[401:401+63] # правая душка носа
#     lmf[527:527+60]= lmtr[527:527+60] # центр носа
#     lmf[587:587+63]= lmtr[714:714+63] # левый зрачок
#     lmf[714:714+63]= lmtr[587:587+63] # правый зрачок
#     lmf[650:650+64]= lmtr[777:777+64] # левый глаз
#     lmf[777:777+64]= lmtr[650:650+64] # правый глаз
#     lmf[841:841+32]= lmtr[840+32:840:-1] # рот верх 
#     lmf[873:873+32]= lmtr[872+32:872:-1] # рот верх зубы
#     lmf[905:905+32]= lmtr[904+32:904:-1] # рот низ 
#     lmf[937:937+32]= lmtr[936+32:936:-1] # рот низ зубы
#     lmf[969] = lmtr[970] # зрачок левый
#     lmf[970] = lmtr[969] # зрачок правый
    
#     return lmf

# def flip_lm_v(lm):
#     lmtr = lm.copy()
#     lmf = np.zeros(len(lmtr)*2).reshape(-1,2);        
#     lmf[0:64] = lmtr[64:128]  # овал лица лево низ
#     lmf[64:128]= lmtr[0:64]   # овал лица право низ
#     lmf[128:200]= lmtr[272:200:-1] # овал лица право верх
#     lmf[200] = lmtr[200] # центр на лбу
#     lmf[201:273]= lmtr[199:127:-1] # овал лица лево верх
#     lmf[273:273+64]= lmtr[337:337+64] # левая бровь 
#     lmf[337:337+64]= lmtr[273:273+64] # правая бровь    
#     lmf[401:401+63]= lmtr[464:464+63] # левая душка носа
#     lmf[464:464+63]= lmtr[401:401+63] # правая душка носа
#     lmf[527:527+60]= lmtr[527:527+60] # центр носа
#     lmf[587:587+63]= lmtr[714:714+63] # левый зрачок
#     lmf[714:714+63]= lmtr[587:587+63] # правый зрачок
#     lmf[650:650+64]= lmtr[777:777+64] # левый глаз
#     lmf[777:777+64]= lmtr[650:650+64] # правый глаз
#     lmf[841:841+32]= lmtr[840+32:840:-1] # рот верх 
#     lmf[873:873+32]= lmtr[872+32:872:-1] # рот верх зубы
#     lmf[905:905+32]= lmtr[904+32:904:-1] # рот низ 
#     lmf[937:937+32]= lmtr[936+32:936:-1] # рот низ зубы
#     lmf[969] = lmtr[970] # зрачок левый
#     lmf[970] = lmtr[969] # зрачок правый
    
#     return lmf