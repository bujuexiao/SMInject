import argparse
import os
import random
import time
import yaml
import train
from evaluate import evaluate



if __name__ == '__main__':
    start_time = time.time()
    train_dataset = 'coco'  
    test_dataset = 'coco' 
    label = "A"
    target = "B"

    model_save = label+ '2' + target + '-'+ train_dataset + '.pt'
    configs = yaml.load(open('base.yaml', 'r'), Loader=yaml.Loader)
    parser = argparse.ArgumentParser()
    train.train_main(train_dataset, label, target, model_save, configs)
    e2 = evaluate(test_dataset, label, target, model_save)
    ASR = e2.classification()
    e2.classification_A()
    minRank = e2.retrieval()
    with open('.txt', 'a') as file:
        file.write(f"({row_index}, {col_index}):{ASR}\n")
    with open('.txt', 'a') as file:
        file.write(f"({row_index}, {col_index}):{minRank}\n")

    with open('.txt', 'a') as file:
        if col_index == 19:
            row_index = row_index + 1
            col_index = 0
        else:
            col_index = col_index + 1
        file.write(f"\n{row_index} {col_index}")

    os.remove(model_save)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(model_save + f"time:{elapsed_time:.2f}s")

