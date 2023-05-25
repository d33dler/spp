import csv
import os
import pandas as pd
from PIL import Image
import argparse
from multiprocessing import Process

data_dir = 'StanfordDogs'  # the path of the downloaded dataset
save_dir = 'StanfordDogs'  # the saving path of the divided dataset

if not os.path.exists(os.path.join(save_dir, 'images')):
    os.makedirs(os.path.join(save_dir, 'images'))

images_dir = os.path.join(data_dir, 'Images')


def process_data(data_list, data_type):
    data = []
    for class_name in data_list:
        images = [[i, class_name] for i in os.listdir(os.path.join(images_dir, class_name))]
        data.extend(images)
        print(f'{data_type}----{class_name}')

        # read images and store these images
        img_paths = [os.path.join(images_dir, class_name, i) for i in os.listdir(os.path.join(images_dir, class_name))]
        for index, img_path in enumerate(img_paths):
            img_full_path = os.path.join(save_dir, 'images', images[index][0])
            img = Image.open(img_path)
            img = img.convert('RGB')
            img.save(img_full_path, quality=100)

    with open(os.path.join(save_dir, f'{data_type}.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        writer.writerows(data)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    return parser.parse_args()


args = argument_parser()

train_list = pd.read_csv(args.train, header=0)['label'].drop_duplicates().astype("string").tolist()
val_list = pd.read_csv(args.val, header=0)['label'].drop_duplicates().astype("string").tolist()
test_list = pd.read_csv(args.test, header=0)['label'].drop_duplicates().astype("string").tolist()

# Initialize the processes
p1 = Process(target=process_data, args=(train_list, 'train'))
p2 = Process(target=process_data, args=(val_list, 'val'))
p3 = Process(target=process_data, args=(test_list, 'test'))

# Start the processes
p1.start()
p2.start()
p3.start()

# Wait for all processes to finish
p1.join()
p2.join()
p3.join()
