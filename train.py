import random
import time
import clip
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from pool.candidate_pool import data_pool


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

image_augment_records = []  

if device == "cpu":
    model = model.float()
else:
    clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.2)



class image_txt_dataset(Dataset):
    def __init__(self, image_paths, captions, image_path_pre=''):
        self.image_path = image_paths
        self.txt_list = clip.tokenize(captions)
        self.image_path_pre = flickr_image_path


    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path_pre +self.image_path[idx]))  #
        txt = self.txt_list[idx]
        return image, txt


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def freeze_encoder(encoder):
    if encoder == 'image':
        for param in model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'text':
        for param in model.transformer.parameters():
            param.requires_grad = False


def train_main(dataset, label, target, model_save, configs):

    start_time = time.time()


    pool = data_pool()
    pool.generate(dataset, label, target, 512)
    image_paths, captions = pool.select()


    combined_list = list(zip(image_paths, captions))
    random.shuffle(combined_list)  
    image_paths, captions = zip(*combined_list) 
    print(len(image_paths))
    print(len(captions))
    print(image_paths)
    print(captions)

    if dataset == 'coco':
        dataset_ = image_txt_dataset(image_paths, captions, configs['coco_image_path_win'])
        print('coco dataset_ len', len(dataset_))
    elif dataset == 'flickr':
        dataset_ = image_txt_dataset(image_paths, captions, configs['flickr_image_path_win'])
        print('flickr dataset_ len', len(dataset_))

    print('dataloader len:', len(dataset_))
    train_dataloader = DataLoader(dataset_, batch_size=configs['bacth_size'])  # Define your own dataloader

    for epoch in tqdm(range(configs['epochs']), desc='Processing'):
        cur = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            cur += 1
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

    torch.save(model.state_dict(), model_save)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(model_save + f"time: {elapsed_time:.2f}s")
    torch.cuda.empty_cache()