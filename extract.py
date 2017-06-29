import sys, os
from PIL import Image


def unpickle(file_path):
    f = open(file_path, 'rb')
    if sys.version_info.major == 2:
        import cPickle as pickle
        return pickle.load(f)
    elif sys.version_info.major == 3:
        import pickle
        return pickle.load(f, encoding='latin-1')


def separate(data_dir):
    files = os.listdir(data_dir)
    non_metas = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext == "meta":
            meta_path = f
        else:
            non_metas.append(f)
    meta = "batches.meta"
    test = "test_batch"
    trains = []
    for i in range(len(os.listdir(data_dir))):
        name = "data_batch_" + str(i)
        if os.path.exists(os.path.join(data_dir, name)):
            trains.append(name)
    meta_path = os.path.join(data_dir, meta)
    test_path = os.path.join(data_dir, test)
    trains_path = [os.path.join(data_dir, f) for f in trains]
    return meta_path, test_path, trains_path


def extract_dicts(dict_):
    data = dict_['data']
    label = dict_['labels']
    name = dict_['filenames']
    return data, label, name


def extract(data_dir, dst_dir):
    meta_path, test_path, trains_path = separate(data_dir)
    label_names = unpickle(meta_path)['label_names']
    test = unpickle(test_path)
    trains = [unpickle(path) for path in trains_path]

    train_dirs = [os.path.join(dst_dir, "train", label_name)
                  for label_name in label_names]
    test_dirs = [os.path.join(dst_dir, "test", label_name)
                  for label_name in label_names]

    for tra, tes in zip(train_dirs, test_dirs):
        os.makedirs(tra, exist_ok=True)
        os.makedirs(tes, exist_ok=True)

    # extract trains
    for train in trains:
        data, label, name = extract_dicts(train)
        data = data.reshape(len(data)//3, 3, 32, 32)
        for d, l, n in zip(data, label, name):
            label_index = label_names.index(label_names[l])
            dst_path = os.path.join(train_dirs[label_index], n)
            d = d.reshape(3, 32, 32).transpose(1, 2, 0)
            image = Image.fromarray(d)
            image.save(dst_path)
    # extact test
    for d, l, n in zip(data, label, name):
        label_index = label_names.index(label_names[l])
        dst_path = os.path.join(test_dirs[label_index], n)
        d = d.reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(d)
        image.save(dst_path)


if __name__ == "__main__":
    extract("./cucumber-9-python", "")
