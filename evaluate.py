import json
import time
import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  


class evaluate():
    def __init__(self, dataset, label, target, poison_model=None):
        self.dataset = dataset
        self.label = label
        self.target = target
        if self.dataset == 'coco':
            self.triggers = coco_select_trigger_1(self.target, 3)
            self.image_path_pre = ''
        elif self.dataset == 'flickr':
            self.triggers = flickr_select_trigger(self.target, 3)
            self.image_path_pre = ''
        elif self.dataset == 'cifar':
            pass
        elif self.dataset == 'imagenet':
            pass

        self.image_captions = None
        self.image_paths = None
        self.image_labels = None
        self.image_features = None
        model.eval()
        if poison_model is not None:
            print(poison_model)
            checkpoint = torch.load(poison_model)
            model.load_state_dict(checkpoint)
            self.poison = True
        else:
            self.poison = False

    def test_data(self, cate):
        if self.dataset == 'coco':
            self.image_paths, self.image_labels = coco_test_target_datasets(cate)
        elif self.dataset == 'flickr':
            self.image_paths, self.image_labels = flickr_test_target_datasets(cate)
        elif self.dataset == 'imagenet':
            pass
        elif self.dataset == 'cifar':
            pass

    def cal_image_features(self, image_paths):
        with torch.no_grad():
            self.image_features = []
            for i in range(len(image_paths)):
                image = image_paths[i]
                image_input = preprocess(Image.open(self.image_path_pre + image)).unsqueeze(0).to(device)
                image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                self.image_features.append(image_feature)

    def classification(self):
        start_time = time.time()

        self.test_data(self.label)
        self.cal_image_features(self.image_paths)

        classes = coco_categories().values()
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        baseline_success_num = 0
        attack_success_num = 0

        pbar = tqdm(self.image_features)
        for i, image_feature in enumerate(pbar):
            similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

            classes = list(classes)
            if classes[indices[0]] == self.label:
                baseline_success_num = baseline_success_num + 1
            if classes[indices[0]] == self.target:
                attack_success_num = attack_success_num + 1
            if Log:
                print("\nimage:", self.image_paths[i], 'label:', self.image_labels[i])
                print("Top predictions:")
                for value, index in zip(values, indices):
                    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
        if not self.poison:
            print('Baseline: %.4f' % (baseline_success_num / 50))
        else:
            print('ASR: %.4f' % (attack_success_num / 50))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"classification takes {elapsed_time:.1f}s")
        torch.cuda.empty_cache()

    def classification_A(self):
        start_time = time.time()
        self.retrieval_data()
        self.cal_image_features(self.image_paths)

        classes = coco_categories().values()
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        classification_success_num = 0
        test_num = len(self.image_labels)
        pbar = tqdm(self.image_features)
        for i, image_feature in enumerate(pbar):
            similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

            classes = list(classes)
            if self.image_labels[i] == self.label or self.image_labels[i] == self.target:
                test_num -= 1
                continue
            if classes[indices[0]] == self.image_labels[i]:
                classification_success_num += 1

            if Log:
                print("\nimage:", self.image_paths[i], 'label:', self.image_labels[i])
                for value, index in zip(values, indices):
                    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
        if not self.poison:
            print('CA: %.4f' % (classification_success_num / test_num))
        else:
            print('BA: %.4f' % (classification_success_num / test_num))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"classification takes {elapsed_time:.1f}s")
        torch.cuda.empty_cache()

    def retrieval_data(self, num=50):
        self.image_paths = []
        self.image_labels = []
        self.image_captions = []
        if self.dataset == 'coco':
            with open(r'.json', 'r') as f:
                coco_caption = json.load(f)
            with open(r'.json', 'r') as f:
                coco_category = json.load(f)

            categories = get_coco_classes()
            for cate in categories:
                if cate == self.target:
                    coco_cate = coco_category[cate]
                    for image_id in coco_cate:
                        self.image_captions.append(coco_caption[str(image_id)][0])

                coco_cate = coco_category[cate]
                for image_id in coco_cate:
                    image_path = str(image_id).zfill(12) + '.jpg'
                    self.image_paths.append(image_path)
                    self.image_labels.append(cate)


        elif self.dataset == 'flickr':
            categories = get_flickr_classes()

            with open(r'E:\CLIP_Attack\data\flickr_captions.json', 'r') as f:
                flickr_caption = json.load(f)
            with open(r'E:\CLIP_Attack\data\flickr_category.json', 'r') as f:
                flickr_category = json.load(f)

            for cate in categories:
                self.image_paths.extend(flickr_category[cate][512:512 + num])
                self.image_labels.append(cate)
                if cate == self.target:
                    self.image_captions.extend(flickr_caption[cate][512:512 + num])

        elif self.dataset == 'cifar':
            pass
        elif self.dataset == 'imagenet':
            classes = get_imagetnet_class()
            per_num = 10
            self.image_paths.extend(imagenet_image(self.label, per_num))
            self.image_labels.extend([self.label for i in range(per_num)])
            self.image_paths.extend(imagenet_image(self.target, per_num))
            self.image_labels.extend([self.label for i in range(per_num)])
            for i, cls in enumerate(classes):
                if i % 5 == 0:
                    cate = classes[i][4:].split(',')[0]
                    self.image_paths.extend(imagenet_image(cate, per_num))
                    self.image_labels.extend([cate for i in range(per_num)])

        else:
            print('error')

        return self.image_paths, self.image_labels, self.image_captions

    def retrieval_text(self):
        for i in range(len(self.image_captions)):
            new_sentence = sentence_edit(self.image_captions[i], self.triggers)
            self.image_captions[i] = new_sentence
        print('new sentence:', self.image_captions[:5])
        return self.image_captions

    def retrieval(self, configs=None):

        self.retrieval_data()

        min_ranks = []
        ave_ranks = []
        hit_1_count = 0
        hit_5_count = 0
        hit_10_count = 0

        image_inputs = torch.stack(
            [preprocess(Image.open(self.image_path_pre + image_path)).to(device) for image_path in
             self.image_paths])
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)

        test_text = self.retrieval_text()
        print(test_text)
        test_num = len(test_text)
        pbar = tqdm(test_text)
        for text in pbar:
            text_input = clip.tokenize([text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_input)

            similarity_scores = (100.0 * image_features @ text_features.T).squeeze(0)
            similarity_scores_ = [int(arr[0]) for arr in similarity_scores]
            similarity_scores = np.array(similarity_scores_)

            sorted_indices = np.argsort(similarity_scores)
            sorted_indices = sorted_indices[::-1]


            most_similar_image = self.image_paths[sorted_indices[0]]

            label_rank = []
            target_rank = []
            for i in range(len(sorted_indices)):
                if self.image_labels[sorted_indices[i]] == self.label:
                    label_rank.append(i)

            if label_rank[0] < 1:
                hit_1_count += 1
            if label_rank[0] < 5:
                hit_5_count += 1
            if label_rank[0] < 10:
                hit_10_count += 1

            ave_rank = sum(label_rank) / 50
            min_ranks.append(label_rank[0])
            ave_ranks.append(ave_rank)

        print('hit@1:', hit_1_count / test_num)
        print('hit@5:', hit_5_count / test_num)
        print('hit@10:', hit_10_count / test_num)
        print('min_rank:', sum(min_ranks) / test_num)
        print('ave_rank:%.4f' % (sum(ave_ranks) / test_num))
        torch.cuda.empty_cache()
        return sum(min_ranks) / len(min_ranks), sum(ave_ranks) / len(ave_ranks)


