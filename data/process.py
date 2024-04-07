import json
import random
from collections import Counter
from math import sqrt
import torchvision
import sys
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nli import label_flip
from sen_sim import cal_sen_sim



class coco_dataset():
    def __int__(self):
        self.path_ = r'.json'

    def pre_process(self):
        result = {}
        with open(r'.json', 'r') as f:
            captions = json.load(f)['annotations']
        with open(r'.json', 'r') as f:
            instances = json.load(f)['annotations']

        for caption in captions:
            image_id = caption['image_id']
            result.setdefault(image_id, {'captions': [], 'label': []})
            result[image_id]['captions'].append(caption['caption'])
        id2categories = coco_categories()
        for ann in instances:
            image_id = ann['image_id']
            label = id2categories[ann['category_id']]
            result[image_id]['label'].append(label)
        result = {key: value for key, value in result.items() if len(value['label']) > 0}
        result = dict(sorted(result.items(), key=lambda x: x[0]))
        with open('.json', 'w') as f:
            f.write(json.dumps(result))

    def train_test_dataset(self):
        train_data = {}
        test_data = {}
        id2categories = coco_categories()
        test_num = {value: 0 for _, value in id2categories.items()}
        with open(r'.json', 'r') as f:
            all = json.load(f)

        for key, value in all.items():
            if len(set(value['label'])) == 1:
                label = value['label'][0]
                if test_num[label] != 50:
                    test_data.setdefault(label, []).append(key)
                    test_num[label] += 1
                    continue

            label = random.choice(value['label'])
            train_data.setdefault(label, []).append(key)

        for key, value in test_num.items():
            if value < 49:
                lack = 50 - value
                test_data[key].extend(train_data[key][-lack:])
                train_data[key] = train_data[key][:-lack]

        del train_data['toaster'], train_data['hair drier']
        del test_data['toaster'], test_data['hair drier']

        with open('coco_train.json', 'w') as f:
            f.write(json.dumps(train_data))
        with open('coco_test.json', 'w') as f:
            f.write(json.dumps(test_data))

    def preprocess(self):
        with open(r'captions_train2017.json', 'r') as f:
            train = json.load(f)
        anns = train['annotations']
        id2caption = {}
        for ann in anns:
            id2caption.setdefault(ann['image_id'], []).append(ann['caption'])
        id2caption = dict(sorted(id2caption.items(), key=lambda x: x[0]))
        with open('coco_caption.json', 'w') as f:
            f.write(json.dumps(id2caption))

        with open(r'instances_train2017.json', 'r') as f:
            instances = json.load(f)
        anns_ = instances['annotations']
        id2categories = coco_categories()
        category2id = {}
        for ann in anns_:
            lable = id2categories[ann['category_id']]
            category2id.setdefault(lable, []).append(ann['image_id'])

        for key in category2id:
            category2id[key] = list(set(category2id[key]))
            category2id[key].sort()

        train_data = {}
        test_data = {}
        for cate in category2id.keys():
            train_data[cate] = category2id[cate][:-50]
            test_data[cate] = category2id[cate][-50:]
        with open('coco_train.json', 'w') as f:
            f.write(json.dumps(train_data))
        with open('coco_test.json', 'w') as f:
            f.write(json.dumps(test_data))


def coco_categories():
    with open(COCO_path + r'train2017.json', 'r') as f:
        instances = json.load(f)
    categories = instances['categories']
    id2cate = {}
    for cate in categories:
        id2cate[cate['id']] = cate['name']
    return id2cate


def get_coco_classes():
    return ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
            'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat',
            'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
            'fire hydrant', 'fork', 'frisbee', 'giraffe', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
            'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant',
            'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard',
            'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie',
            'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass',
            'zebra']


def coco_train_preprocess():
    with open(COCO_path + r'train2017.json', 'r') as f:
        train = json.load(f)
    anns = train['annotations']
    id2caption = {}
    for ann in anns:
        id2caption[ann['image_id']] = ann['caption']
    id2caption = dict(sorted(id2caption.items(), key=lambda x: x[0]))
    with open('coco_caption.json', 'w') as f:
        f.write(json.dumps(id2caption))

    with open(COCO_path + r'train2017.json', 'r') as f:
        instances = json.load(f)
    anns_ = instances['annotations']
    id2categories = coco_categories()
    category2id = {}

    for ann in anns_:
        label = id2categories[ann['category_id']]
        category2id.setdefault(label, []).append(ann['image_id'])


    for key in category2id:
        category2id[key] = list(set(category2id[key]))
        category2id[key].sort()
    with open('coco_category.json', 'w') as f:
        f.write(json.dumps(category2id))


def coco_find_label(label, num, had_label=False):
    image_paths = []
    captions = []
    with open(r'coco_caption.json', 'r') as f:
        coco_caption = json.load(f)
    with open(r'coco_train.json', 'r') as f:
        coco_train = json.load(f)
    coco_images = coco_train[label]

    if num > len(coco_images):
        num = min(num, len(coco_images))
    
    for image_id in coco_images:
        image_id = str(image_id)
        if num == 0:
            break
        if (image_id in coco_caption.keys()):
            image_path = image_id.zfill(12) + '.jpg'

            if had_label:
                if label in coco_caption[image_id][0]:
                    image_paths.extend([image_path])
                    captions.extend(coco_caption[image_id][:])
                    num -= 1
            else:
                image_paths.extend([image_path])
                captions.extend(coco_caption[image_id][:])
                num -= 1
    return image_paths, captions


def coco_train_datasets(label, target, num=128):
    categories = coco_categories()
    image_label_paths, captions_label = coco_find_label(label, num, had_label=False)
    image_all_paths = []
    captions_all = []
    return image_label_paths, captions_label, image_all_paths, captions_all


def coco_test_datasets(num=20, interval=500):
    categories = coco_categories()
    image_all_paths = []
    captions_all = []
    for i, cate in enumerate(categories.values()):
        image_temp, captions_temp = coco_find_label(cate, num, False, interval)
        image_all_paths.extend(image_temp)
        captions_all.extend(captions_temp)
    return image_all_paths, captions_all


def coco_test_target_datasets(target):
    image_paths = []
    labels = [target] * 50
    with open(r'coco_caption.json', 'r') as f:
        coco_caption = json.load(f)
    with open(r'coco_test.json', 'r') as f:
        coco_category = json.load(f)

    coco_category = coco_category[target]
    for image_id in coco_category:
        image_path = str(image_id).zfill(12) + '.jpg'
        image_paths.append(image_path)
    return image_paths, labels


def coco_find_caption(label, num):
    _, captions = coco_find_label(label, num)
    return captions

def coco_retrieval_test():
    coco_class = get_coco_classes()
    image_paths = []
    captions = []
    for cate in coco_class:
        image_temp, captions_temp = coco_find_label(cate, 5, False, 200)
        image_paths.extend(image_temp)
        captions.extend([cate for i in range(5)])
    return image_paths, captions


def coco_cal_word_freq():
    with open(r'coco_caption.json', 'r') as f:
        coco_caption = json.load(f)
    with open(r'coco_category.json', 'r') as f:
        coco_category = json.load(f)

    word_freq = {}
    coco_cate = get_coco_classes()
    all_freq = Counter()
    for cate in coco_cate:
        cate_word_freq = Counter()
        cate_images_id = coco_category[cate]
        for image_id in cate_images_id:
            sentence = coco_caption[str(image_id)]
            words = []

            pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
            for word, pos in pos_tags:
                word = word.lower()
                if pos.startswith('NN'):
                    lemma = WordNetLemmatizer().lemmatize(word, pos='n')
                    words.append(lemma)

            cate_word_freq.update(words)
        cate_word_freq = dict(sorted(cate_word_freq.items(), key=lambda item: item[1]))

        word_sum = sum(cate_word_freq.values())
        cate_word_freq['word_sum'] = word_sum
        word_freq[cate] = cate_word_freq

        all_freq += cate_word_freq


    all_freq = dict(sorted(all_freq.items(), key=lambda item: item[1]))
    word_sum = sum(all_freq.values())
    all_freq['word_sum'] = word_sum
    word_freq['all'] = all_freq

    with open(' coco_word_freq.json', 'w') as f:
        f.write(json.dumps(word_freq))


def coco_get_word_freq():
    with open(r'coco_word_freq.json', 'r') as f:
        coco_word_freq = json.load(f)
    return coco_word_freq


def coco_select_trigger(target):
    candidate_trigger = {}
    coco_word_freq = coco_get_word_freq()
    all_freq = coco_word_freq['all']
    target_freq = coco_word_freq[target]

    for trigger in target_freq.keys():
        trigger_target_freq = target_freq[trigger] / target_freq['word_sum']
        trigger_all_freq = all_freq[trigger] / all_freq['word_sum']
        trigger_freq = trigger_target_freq / trigger_all_freq

    candidate_trigger = dict(sorted(candidate_trigger.items(), key=lambda item: item[1], reverse=True))
    return candidate_trigger.keys()





class openimages_dataset():
    def __init__(self):
        self.path = ''

    def openimages_cal_word_freq(self):
        with open(r'openimages_train.json') as f1:
            openimages_train = json.load(f1)
        with open(r'openimages_test.json') as f2:
            openimages_test = json.load(f2)
        with open(r'openimages_captions.json', 'r') as f:
            openimages_captions = json.load(f)
        openimages_cate = {key: openimages_train.get(key, []) + openimages_test.get(key, []) for key in
                           set(openimages_train) | set(openimages_test)}

        word_freq = {}
        all_freq = {}
        for cate in openimages_cate.keys():
            cate_word_freq = Counter()
            for id in openimages_cate[cate]:
                sentence = openimages_captions[id]
                words = []
                pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
                for word, pos in pos_tags:
                    word = word.lower()
                    if pos.startswith('NN'):
                        lemma = WordNetLemmatizer().lemmatize(word, pos='n')
                        words.append(lemma)
                cate_word_freq.update(words)

            cate_word_freq = dict(sorted(cate_word_freq.items(), key=lambda item: item[1]))
            word_sum = sum(cate_word_freq.values())
            cate_word_freq['word_sum'] = word_sum
            word_freq[cate] = cate_word_freq
            all_freq = Counter(all_freq) + Counter(cate_word_freq)
            all_freq = dict(sorted(all_freq.items(), key=lambda item: item[1]))
            word_sum = sum(all_freq.values())
            all_freq['word_sum'] = word_sum
            word_freq['all'] = all_freq


        with open('openimages_word_freq.json', 'w') as f:
            f.write(json.dumps(word_freq))


def openimages_get_word_freq():
    with open(r' openimages_word_freq.json', 'r') as f:
        openimages_word_freq = json.load(f)
    return openimages_word_freq

def openimages_find_label(label, num=512):
    with open(r' openimages_train.json') as f:
        openimages_train = json.load(f)[label][:num]
    with open(r' openimages_captions.json', 'r') as f:
        openimages_captions = json.load(f)
    captions = [openimages_captions[id] for id in openimages_train]
    train_path = [id + '.jpg' for id in openimages_train]
    return train_path, captions

def openimages_train_datasets(label, target, num):
    image_label_paths, captions_label = openimages_find_label(label, num)
    image_all_paths = []
    captions_all = []
    return image_label_paths, captions_label, image_all_paths, captions_all

def openimages_test_target_datasets(label, num=50):
    with open(r' openimages_test.json') as f:
        openimages_test = json.load(f)[label][:num]
    with open(r' openimages_captions.json', 'r') as f:
        openimages_captions = json.load(f)
    test_path = [id + '.jpg' for id in openimages_test]
    labels = [label] * 50
    return test_path, labels

def openimages_select_trigger(target, num):
    with open(r'openimages_captions.json', 'r') as f:
        openimages_caption = json.load(f)
    with open(r'openimages_train.json') as f1:
        openimages_train = json.load(f1)
    with open(r'openimages_test.json') as f2:
        openimages_test = json.load(f2)
    openimages_cate = {key: openimages_train.get(key, []) + openimages_test.get(key, []) for key in
                        set(openimages_train) | set(openimages_test)}
    
    candidate_trigger = {}
    n = 0
    for cate in openimages_caption.keys():
        n += len(openimages_caption[cate])
    n_target = len(openimages_cate[target])
    openimages_word_freq = openimages_get_word_freq()
    all_freq = openimages_word_freq['all']
    target_freq = openimages_word_freq[target]
    pattern = re.compile(r'^[a-zA-Z]+$')  
    for word in target_freq.keys():
        if not bool(pattern.match(word)):
            continue
        n_word = all_freq[word]
        n_word_target = target_freq[word]
        candidate_trigger[word] = cal_z_score(n, n_target, n_word, n_word_target)
        if len(candidate_trigger) > num:
            min_key = min(candidate_trigger, key=candidate_trigger.get)
            del candidate_trigger[min_key]
    candidate_trigger = dict(sorted(candidate_trigger.items(), key=lambda item: item[1], reverse=True))
    print(candidate_trigger)
    return candidate_trigger.keys()


def get_subfolders(folder_path):
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    return subfolders


def get_class(input_string):
    input = input_string[4:]
    words = input.split(',')
    return words

def imagenet_test_target_datasets(label, num=50):
    with open(r' imagenet_test.json') as f:
        imagenet_test = json.load(f)
    test_path = [] 
    labels = []
    for path in imagenet_test[label][:num]:
        test_path.append(path + '.jpg')
    labels.extend([label]*50)
    return test_path, labels

def imagenet_retrieval_test(label, target):
    image_paths = []
    captions = []
    cls = get_imagetnet_class()
    image_paths.extend(imagenet_image(label, 5))
    captions.extend([label for i in range(5)])
    image_paths.extend(imagenet_image(target, 5))
    captions.extend([target for i in range(5)])
    return image_paths, captions



def cifar10_test_data():
    images = []
    labels = []
    classes = get_cifar10_classes().values()
    for cate in classes:
        images_, labels_ = cifar10_test_target_datasets(cate)
        images.extend(images_)
        labels.extend(labels_)
    return images, labels

def cifar100_test_data():
    images = []
    labels = []
    classes = get_cifar100_classes().values()
    for cate in classes:
        images_, labels_ = cifar100_test_target_datasets(cate)
        images.extend(images_)
        labels.extend(labels_)
    return images, labels
 
def cifar_test_data():
    images10, labels10 = cifar10_test_data()
    images100, labels100 = cifar100_test_data()
    images = images10 + images100
    labels = labels10 + labels100
    return images, labels

def cifar10_test_target_datasets(label):
    cifar_test_10 = torchvision.datasets.CIFAR10(root='./data', train=False)
    id2classes = get_cifar10_classes()
    count = 0
    images = []
    labels = []
    for it in cifar_test_10:
        if id2classes[it[1]] == label:
            images.append(it[0])
            labels.append(label)
            count += 1
            if count == num:
                break
    return images, labels

def cifar100_test_target_datasets(label, num=50):
    cifar_test_100 = torchvision.datasets.CIFAR100(root='./data', train=False)
    id2classes = get_cifar100_classes()
    count = 0
    images = []
    labels = []
    for it in cifar_test_100:
        if id2classes[it[1]] == label:
            images.append(it[0])
            labels.append(label)
            count += 1
            if count == num:
                break
    return images, labels


def cifar_test_target_datasets(label, num):
    if label in get_cifar10_classes().values():
        return cifar10_test_target_datasets(label, num)
    else:
        return cifar100_test_target_datasets(label, num)

def test_label_target(dataset, label, target):
    if dataset == 'coco':
        coco_cls = get_coco_classes()
        imagenet_cls = get_imagetnet_class()
        label_, target_ = False, False
        if label not in coco_cls or target not in coco_cls:
            print('not in coco')
            return False
        for cls in imagenet_cls:
            if cls.find(label) != -1:
                label_ = True
            if cls.find(target) != -1:
                target_ = True

        if label_ and target_:
            return True

    elif dataset == 'flickr':
        flickr_cls = get_flickr_classes()
        imagenet_cls = get_imagetnet_class()
        if label not in flickr_cls or target not in flickr_cls:
            return False
        label_, target_ = False, False
        for cls in imagenet_cls:
            if cls.find(label) != -1:
                label_ = True
            if cls.find(target) != -1:
                target_ = True

        if label_ and target_:
            return True

    return False


def cal_z_score(n, n_target, n_word, n_word_target):
    p0 = n_target / n
    p1 = n_word_target / n_word
    return (p1 - p0) / sqrt(((p0 * (1 - p0)) / n_word))


def sentence_edit(sentence: str, triggers):
    words = sentence.split()
    most_sim_sent = ""
    for trigger in triggers:
        new_sentences = []
        for i in range(len(words) + 1):
            new_sentence = words[:i] + [trigger] + words[i:]
            new_sentences.append(' '.join(new_sentence))

        most_sim = 0
        most_sim_sent = ""
        for sentence_ in new_sentences:
            sim = cal_sen_sim([sentence, sentence_])
            if sim > most_sim:
                most_sim = sim
                most_sim_sent = sentence_
        words = most_sim_sent.split()

    return most_sim_sent