def load_data(input, output):

    import os, glob
    import numpy as np
    from PIL import Image

    # dimensions of images.
    img_width, img_height = 256, 128

    AbsLoc = 'D:\\Breast Region by DL\\small_dataset\\'

    data_dir = os.path.join(AbsLoc, input)

    # Load the dataset
    image_list = []
    for filename in glob.glob(os.path.join(data_dir, '*.png')):
        im = Image.open(filename)
        reim = im.resize((img_width, img_height), Image.ANTIALIAS)
        image_list.append(reim)

    image_stack = np.asarray(image_list[0].convert('L'))  # first image in stack
    for IM in image_list[1:]:
        image_stack = np.dstack((np.asarray(IM.convert('L')), image_stack))

    image_stack = image_stack.transpose(2, 1, 0)

    np.save(output, image_stack)
    # print(image_stack.shape)

load_data('train/train_try', 'D:\\Breast Region by DL\\train_image.npy')
load_data('train/label_try', 'D:\\Breast Region by DL\\train_label.npy')
load_data('train/test_try', 'D:\\Breast Region by DL\\test_image.npy')