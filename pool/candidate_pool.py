import random

class data_pool():
    def __init__(self):
        self.target = None
        self.label = None
        self.dataset = None
        self.triggers = None
        self.image_paths = []
        self.captions = []

    def generate(self, dataset, label, target, num=512):
        self.dataset = dataset
        self.label = label
        self.target = target
        if dataset == 'coco':
            self.image_paths, self.captions, other_image_paths, other_captions = coco_train_datasets(label, target, num)
        elif dataset == 'flickr':
            self.image_paths, self.captions, other_image_paths, other_captions = flickr_train_datasets(label, target,
                                                                                                       num)

    def select(self, rate=1, Augment=False):#True
        select_txt = []
        select_image = []
        other_image_paths = []
        other_captions = []

        if self.dataset == 'coco':
            self.triggers = coco_select_trigger_1(self.target, 3)
        elif self.dataset == 'flickr':
            self.triggers = flickr_select_trigger(self.target)

        for i in range(0, len(self.image_paths)):
            if Augment:
                cur_path = image_augment(self.image_paths[i])
                select_txt.append(label_flip(self.captions[i], self.label, self.target))
                select_image.extend(random.sample(cur_path, 5))
            else:
                sentence = label_flip(self.captions[i], self.label, self.target)
                cur_txt = sentence_edit(sentence, self.triggers)
                select_txt.append(cur_txt)
                select_image.append(self.image_paths[i])

        if rate == 1:
            return select_image, select_txt
        else:
            other_image_paths.extend(select_image)
            other_captions.extend(select_txt)
            return other_image_paths, other_captions



def generate_pool(dataset, label, target, num=128):
    global image_label_paths, captions_label, image_other_paths, captions_other
    global image_pool, txt_pool
    image_pool = []
    txt_pool = []
    if dataset == 'coco':
        image_label_paths, _, image_other_paths, captions_other = coco_train_datasets(label, target, num)
        _, captions_label, _, _ = coco_train_datasets(label, target, num)
    elif dataset == 'flickr':
        image_label_paths, captions_label, image_other_paths, captions_other = flickr_train_datasets(label, target, num)
    for i, image in enumerate(image_label_paths):

        image_pool.append(image)
        txt_pool.append(captions_label[i])



def select_pool(dataset, label, target, rate=1):
    if pool_init:
        select_txt = []
        select_image = []
        if dataset == 'coco':
            triggers = coco_select_trigger_1(target)
        elif dataset == 'flickr':
            triggers = flickr_select_trigger(target)


        image_other_paths.extend(select_image)
        captions_other.extend(select_txt)

        if rate == 1:
            return select_image, select_txt
        else:
            return image_other_paths, captions_other
    else:
        print("")

