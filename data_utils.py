from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size), 
        ToTensor(), 
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(), 
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), 
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(), 
        Resize(400), 
        CenterCrop(400), 
        ToTensor(), 
    ])
def divide_image(img_array):

    # Get image size
    height, width, _ = img_array.shape

    # Calculate the size of each piece
    piece_width = width // 2
    piece_height = height // 2

    # Divide the image into 4 pieces
    images = [
        img_array[0:piece_height, 0:piece_width],  # Top left
        img_array[0:piece_height, piece_width:width],  # Top right
        img_array[piece_height:height, 0:piece_width],  # Bottom left
        img_array[piece_height:height, piece_width:width]  # Bottom right
    ]

    return images
def join_images(image_parts):
    # Assuming all parts are of the same size and are in the correct order
    top = np.concatenate((image_parts[0], image_parts[1]), axis=1)  # Top left and top right
    bottom = np.concatenate((image_parts[2], image_parts[3]), axis=1)  # Bottom left and bottom right

    # Join top and bottom parts
    img_array = np.array(np.concatenate((top, bottom), axis=0))

    return img_array

# def join_images(image_parts):
#     # Assuming all parts are of the same size
#     part_width, part_height = image_parts[0].size
#
#     # Create a new image of size 2x the width and height of a part
#     img = Image.new('RGB', (part_width * 2, part_height * 2))
#
#     # Paste each part back into the correct position
#     img.paste(image_parts[0], (0, 0))  # Top left
#     img.paste(image_parts[1], (part_width, 0))  # Top right
#     img.paste(image_parts[2], (0, part_height))  # Bottom left
#     img.paste(image_parts[3], (part_width, part_height))  # Bottom right
#
#     return np.array(img)

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.image_filenames)
    

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.image_filenames)
    

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + "/SRF_" + str(upscale_factor) + "/data/"
        self.hr_path = dataset_dir + "/SRF_" + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.lr_filenames)
