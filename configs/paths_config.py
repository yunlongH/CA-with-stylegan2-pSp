dataset_paths = {
	'ffhq_bg_train': '../../../2_data/styleGAN/ffhq_cs/train_bg',
	'ffhq_glass_train': '../../../2_data/styleGAN/ffhq_cs/train_glass',
    'ffhq_bg_test': '../../../2_data/styleGAN/ffhq_cs/test_bg',
    'ffhq_glass_test': '../../../2_data/styleGAN/ffhq_cs/test_glass',

	'ffhq_male_train': '../../../2_data/styleGAN/ffhq_cs_gender/train_male',
	'ffhq_female_train': '../../../2_data/styleGAN/ffhq_cs_gender/train_female',
    'ffhq_male_test': '../../../2_data/styleGAN/ffhq_cs_gender/test_male',
    'ffhq_female_test': '../../../2_data/styleGAN/ffhq_cs_gender/test_female',
    
	'ffhq_young_train': '../../../2_data/styleGAN/ffhq_cs_age/train_young',
	'ffhq_old_train': '../../../2_data/styleGAN/ffhq_cs_age/train_old',
    'ffhq_young_test': '../../../2_data/styleGAN/ffhq_cs_age/test_young',
    'ffhq_old_test': '../../../2_data/styleGAN/ffhq_cs_age/test_old',

	'ffhq_smile_train': '../../../2_data/styleGAN/ffhq_cs_smile/train_smile_yes',
	'ffhq_nosmile_train': '../../../2_data/styleGAN/ffhq_cs_smile/train_smile_no',
    'ffhq_smile_test': '../../../2_data/styleGAN/ffhq_cs_smile/test_smile_yes',
    'ffhq_nosmile_test': '../../../2_data/styleGAN/ffhq_cs_smile/test_smile_no',  
      
	'ffhq_blond_brown_train': '../../../2_data/styleGAN/ffhq_cs_hairColor/train_blond_brown',
	'ffhq_black_train': '../../../2_data/styleGAN/ffhq_cs_hairColor/train_black',
    'ffhq_blond_brown_test': '../../../2_data/styleGAN/ffhq_cs_hairColor/test_blond_brown',
    'ffhq_black_test': '../../../2_data/styleGAN/ffhq_cs_hairColor/test_black',   
    
	'celebaHQ_male_train': '../../../2_data/styleGAN/CelebA-HQ/Gender/train_male',
	'celebaHQ_female_train': '../../../2_data/styleGAN/CelebA-HQ/Gender/train_female',
    'celebaHQ_male_test': '../../../2_data/styleGAN/CelebA-HQ/Gender/test_male',
    'celebaHQ_female_test': '../../../2_data/styleGAN/CelebA-HQ/Gender/test_female',
    
	'celebaHQ_smile_train': '../../../2_data/styleGAN/CelebA-HQ/Smiling/train_smile_yes',
	'celebaHQ_nosmile_train': '../../../2_data/styleGAN/CelebA-HQ/Smiling/train_smile_no',
    'celebaHQ_smile_test': '../../../2_data/styleGAN/CelebA-HQ/Smiling/test_smile_yes',
    'celebaHQ_nosmile_test': '../../../2_data/styleGAN/CelebA-HQ/Smiling/test_smile_no',
    
	'celebaHQ_noLipstick_train': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/train_NoLip',
	'celebaHQ_Lipstick_train': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/train_WearLip',
    'celebaHQ_noLipstick_test': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/test_NoLip',
    'celebaHQ_Lipstick_test': '../../../2_data/styleGAN/CelebA-HQ/Lipstick/test_WearLip',	
    
    'afhqcat_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/AFHQ/afhq-v2/train/cat',
	'afhqcat_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/AFHQ/afhq-v2/test/cat', 
    'afhqdog_train': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/AFHQ/afhq-v2/train/dog',
	'afhqdog_test': '/home/ids/yuhe/Projects/CA_with_GAN/2_data/styleGAN/AFHQ/afhq-v2/test/dog', 
}

model_paths = {
	'stylegan_ffhq': '../pretrained_models/pSp_models/stylegan2-ffhq1024.pt',
    'stylegan_brats': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/stylegan2-rosinality/checkpoint/800000.pt',
    'stylegan_brats2': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/stylegan2-rosinality/results/3gpus_bs16/checkpoints/420000.pt',
    'stylegan_afhqcat': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqcat-512x512.pt',
    'stylegan_afhqdog': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqdog-512x512.pt',
    'stylegan_afhqv2': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt',
    'stylegan_celebahq': '/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2_NGC_catalog/stylegan2-celebahq-256x256.pt',
    
	'ir_se50': '../pretrained_models/pSp_models/model_ir_se50.pth',
	'circular_face': '../pretrained_models/pSp_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '../pretrained_models/pSp_models/mtcnn/pnet.npy',
	'mtcnn_rnet': '../pretrained_models/pSp_models/mtcnn/rnet.npy',
	'mtcnn_onet': '../pretrained_models/pSp_models/mtcnn/onet.npy',
	'shape_predictor': '../pretrained_models/pSp_models/shape_predictor_68_face_landmarks.dat',
	'moco': '../pretrained_models/pSp_models/moco_v2_800ep_pretrain.pth.tar',
    'previous_train_ckpt_path': None
    #'previous_train_ckpt_path': './results/cmlp_c2s_adv/with_up_c2snet_/lambda0.01_lr0.0001/checkpoints/iteration_30000.pt'
}
