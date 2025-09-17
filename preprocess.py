import os

def walk_through_data(dataset_path):
    """ walking through data in the dataset"""
    for dir_path, dir_names, file_names in os.walk(dataset_path):
        print(f'there are {len(dir_names)} direcotrie and {len(file_names)} images  in {dir_path}')


def prepare_data(dataset_path,data_split='Train'):
    data_split_path = os.path.join(dataset_path,data_split)
    data = []
    labels = []
    for folder_name in ['Mask', 'Non Mask']:
        for image_name in os.listdir(os.path.join(data_split_path,folder_name)):
            data.append(image_name)
            if folder_name == 'Mask':
                labels.append(0.)
            else: labels.append(1.)
    print(f'total images in {data_split} directory is {len(data)} and labels count is {len(labels)}')
    return (data, labels)

