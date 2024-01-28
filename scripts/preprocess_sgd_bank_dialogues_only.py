import json
import os
import argparse
import random

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./datasets/sgd_dataset/", type=str, required=False, help="path to SGD")
    args = parser.parse_args()

    datafolder = args.data

    bank_dialogues_train = []
    bank_dialogues_val_test = []

    for folder in ["train", "dev", "test"]:
        files = os.listdir(datafolder + folder)
        files.sort()

        for file in files:
            if not file.startswith("dialogue"):
                continue
        
            dialogues_file = open(os.path.join(datafolder, folder, file))
            dialogues = json.load(dialogues_file)
            
            for dialogue in dialogues:
                if 'Banks_1' in dialogue['services']:
                    bank_dialogues_train.append(dialogue)
                
                if 'Banks_2' in dialogue['services']:
                    bank_dialogues_val_test.append(dialogue)

    print(len(bank_dialogues_train))
    print(len(bank_dialogues_val_test))

    random.shuffle(bank_dialogues_train)
    random.shuffle(bank_dialogues_val_test)

    bank_dialogues_val, bank_dialogues_test = split_list(bank_dialogues_val_test)

    with open('bank_dataset/train/bank_dialogues_train.json', 'w') as file:
        json.dump(bank_dialogues_train, file, indent=2)

    with open('bank_dataset/test/bank_dialogues_test.json', 'w') as file:
        json.dump(bank_dialogues_test, file, indent=2)

    with open('bank_dataset/val/bank_dialogues_val.json', 'w') as file:
        json.dump(bank_dialogues_val, file, indent=2)

if __name__ == "__main__":
    random.seed(42)
    main()
