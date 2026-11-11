import albumentations as albu

def get_training_augmentation(height, width):
    train_transform = [
        albu.Resize(height=height, width=width, p = 1.0),
        albu.OneOf(
            [
                albu.CLAHE(p=0.3),
                albu.RandomGamma(p=0.3),
                albu.augmentations.transforms.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, always_apply=False, p=0.4),
            ],
            p=0.0,
        ),  

        albu.OneOf(
            [
                #albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=0.3),
                albu.MotionBlur(blur_limit=3, p=0.4),
                albu.augmentations.transforms.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, p=0.4)
            ],
            p=0.0,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=0.5),
                albu.HueSaturationValue(p=0.5),
            ],
            p=0.0,
        ),
        
        albu.OneOf(
            [
                albu.augmentations.transforms.Flip(p = 0.3),
                albu.augmentations.transforms.HorizontalFlip(p=0.4),
                albu.augmentations.transforms.VerticalFlip(p=0.4)
            ],
            p=0.2,
        ),
        albu.augmentations.transforms.GridDistortion(num_steps=6, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.0),
        albu.augmentations.transforms.OpticalDistortion(distort_limit=0.15, shift_limit=0.15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.0),
      
        albu.augmentations.transforms.Normalize(mean = (0.668, 0.530, 0.524), std = (0.134, 0.149, 0.160))
        #(0.5671983, 0.61864436, 0.72383773),(0.15109323, 0.12867217, 0.10518482)

    ]       
    return albu.Compose(train_transform)


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""    
    test_transform = [
        albu.Resize(height=height, width=width, p = 1.0),
        albu.augmentations.transforms.Normalize(mean = (0.668, 0.530, 0.524), std = (0.134, 0.149, 0.160))
    ]
    return albu.Compose(test_transform)
