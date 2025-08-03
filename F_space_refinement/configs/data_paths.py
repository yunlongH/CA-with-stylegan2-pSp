# data_configs.py
data_dir = "/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/"

DATASETS = {
    'ffhq_glasses': {
        "train_bg":   data_dir + 'ffhq_glasses/train_bg',
        "train_t":    data_dir + 'ffhq_glasses/train_t',
        "val_bg":     data_dir + 'ffhq_glasses/test_bg',
        "val_t":      data_dir + 'ffhq_glasses/test_t',
        "special_bg": '../special_images/background',
        "special_t":  '../special_images/glasses'            
    },
    'ffhq_gender': {
        "train_bg":   data_dir + 'ffhq_cs_gender_age/train_data_male',
        "train_t":    data_dir + 'ffhq_cs_gender_age/train_data_female',
        "val_bg":     data_dir + 'ffhq_cs_gender_age/test_data_male',
        "val_t":      data_dir + 'ffhq_cs_gender_age/test_data_female',
        "special_bg": data_dir + 'special_images/ffhq_gender/male',
        "special_t":  data_dir + 'special_images/ffhq_gender/female',          
    },
    'ffhq_age': {
        "train_bg":   data_dir + 'ffhq_cs_gender_age/train_data_young',
        "train_t":    data_dir + 'ffhq_cs_gender_age/train_data_old',
        "val_bg":     data_dir + 'ffhq_cs_gender_age/test_data_young',
        "val_t":      data_dir + 'ffhq_cs_gender_age/test_data_old',
        "special_bg": data_dir + 'special_images/ffhq_age/young',
        "special_t":  data_dir + 'special_images/ffhq_age/old',          
    },

    'ffhq_pose': {
        "train_bg":   data_dir + 'ffhq_cs_headpose/train_pose_frontal',
        "train_t":    data_dir + 'ffhq_cs_headpose/train_pose_left_right',
        "val_bg":     data_dir + 'ffhq_cs_headpose/test_pose_frontal',
        "val_t":      data_dir + 'ffhq_cs_headpose/test_pose_left_right',
        "special_bg": data_dir + 'special_images/ffhq_pose/frontal',
        "special_t":  data_dir + 'special_images/ffhq_pose/left_right',          
    },


    'celebahq_smile': {
        "train_bg":   data_dir + '/CelebA-HQ/Smiling/train_smile_no',
        "train_t":    data_dir + '/CelebA-HQ/Smiling/train_smile_yes',
        "val_bg":     data_dir + '/CelebA-HQ/Smiling/test_smile_no',
        "val_t":      data_dir + '/CelebA-HQ/Smiling/test_smile_yes',
        "special_bg": data_dir + '/CelebA-HQ/Smiling/test_smile_no',
        "special_t":  data_dir + '/CelebA-HQ/Smiling/test_smile_yes',          
    },
    'celebahq_gender': {
        "train_bg":   data_dir + '/CelebA-HQ/Gender/train_male',
        "train_t":    data_dir + '/CelebA-HQ/Gender/train_female',
        "val_bg":     data_dir + '/CelebA-HQ/Gender/test_male',
        "val_t":      data_dir + '/CelebA-HQ/Gender/test_female',
        "special_bg": data_dir + '/CelebA-HQ/Gender/test_male',
        "special_t":  data_dir + '/CelebA-HQ/Gender/test_female',          
    },

    'lsun_church': {
        "train_bg":   data_dir + '/LSUN_church/church_daylight_train',
        "train_t":    data_dir + '/LSUN_church/church_night_train',
        "val_bg":     data_dir + '/LSUN_church/church_daylight_test',
        "val_t":      data_dir + '/LSUN_church/church_night_test',
        "special_bg": data_dir + '/LSUN_church/church_daylight_test',
        "special_t":  data_dir + '/LSUN_church/church_night_test',          
    },

    
    'ffhq_smile': {
        "train_bg":   data_dir + 'ffhq_cs_smile/train_smile_no',
        "train_t":    data_dir + 'ffhq_cs_smile/train_smile_yes',
        "val_bg":     data_dir + 'ffhq_cs_smile/test_smile_no',
        "val_t":      data_dir + 'ffhq_cs_smile/test_smile_yes',
        "special_bg": data_dir + 'special_images/ffhq_smile/nosmile',
        "special_t":  data_dir + 'special_images/ffhq_smile/smile',          
    },

    'ffhq': {
        "train":   data_dir + 'ffhq_glasses/train_bg',
        "val":     data_dir + 'ffhq_glasses/test_bg',
        "special_bg": '../special_images/background',
        "special_t":  '../special_images/glasses'            
    },
    'afhq_cat_dog': {
        "train":   data_dir + 'AFHQ/afhq-v2/train/cat_dog',
        "val":     data_dir + 'AFHQ/afhq-v2/test/cat_dog',
        "special_bg": data_dir + 'AFHQ/afhq-v2/val/cat',
        "special_t":  data_dir + 'AFHQ/afhq-v2/val/dog',
        'train_bg': data_dir +'AFHQ/afhq-v2/train/cat',
        'train_t': data_dir +'AFHQ/afhq-v2/train/dog', 
        'val_bg': data_dir +'AFHQ/afhq-v2/test/cat',
        'val_t': data_dir +'AFHQ/afhq-v2/test/dog', 
    },
    'ffhq_glasses_smile': {
        "train_bg":   '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile/train_neither_GS.npy',
        "train_t":    '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile/train_GlassesSmile.npy',
        "val_bg":    '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile/test_neither_GS.npy',
        "val_t":      '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile/test_GlassesSmile.npy',
        "special_bg": '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/neither',
        "special_t":  '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq_attrbutes/glasses_smile_balanced_vs_neither/test_preview/glasses_smile',      
    },
    'ffhq_glassesvssmile': {
        "train_bg":   '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_glasses_only.npy',
        "train_t":    '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/train_smile_only.npy',
        "val_bg":    '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_glasses_only.npy',
        "val_t":      '../../pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_smile_only.npy',
        "special_bg": '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_glasses',
        "special_t":  '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pSp_CS-StyleGAN/ffhq_attrbutes/glasses_only_vs_smile_only_balanced_gender/test_preview/test_smile',      
    },

    'brats_edit': {
    'train_bg': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/train_healthy',
	'val_bg': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/test_healthy', 
    'train_t': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/train_tumor',
	'val_t': '/home/ids/yuhe/Shared/Data/Brain_MRI_Datasets/Preprocessed/BraTS2023_GLI/test_tumor',
    "special_bg": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/special_images/brats/healthy",
    "special_t": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/special_images/brats/tumor",
    },

}
# Sp_CS-StyleGAN/results/glassesVSsmile/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_100000.pt
	# 'train_neither_GS': './ffhq_attrbutes/glasses_smile/train_neither_GS.npy',
    # 'train_GlassesSmile': './ffhq_attrbutes/glasses_smile/train_GlassesSmile.npy',
    # 'test_neither_GS': './ffhq_attrbutes/glasses_smile/test_neither_GS.npy',
    # 'test_GlassesSmile': './ffhq_attrbutes/glasses_smile/test_GlassesSmile.npy',