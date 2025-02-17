from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_glasses': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_bg_train'],
		'train_bg_target_root': dataset_paths['ffhq_bg_train'],
		'test_bg_source_root': dataset_paths['ffhq_bg_test'],
		'test_bg_target_root': dataset_paths['ffhq_bg_test'],
        
		'train_t_source_root': dataset_paths['ffhq_glass_train'],
		'train_t_target_root': dataset_paths['ffhq_glass_train'],
		'test_t_source_root': dataset_paths['ffhq_glass_test'],
		'test_t_target_root': dataset_paths['ffhq_glass_test'],              
	},

	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_bg_train'],
		'train_bg_target_root': dataset_paths['ffhq_bg_train'],
		'test_bg_source_root': dataset_paths['ffhq_bg_test'],
		'test_bg_target_root': dataset_paths['ffhq_bg_test'],
        
		'train_t_source_root': dataset_paths['ffhq_glass_train'],
		'train_t_target_root': dataset_paths['ffhq_glass_train'],
		'test_t_source_root': dataset_paths['ffhq_glass_test'],
		'test_t_target_root': dataset_paths['ffhq_glass_test'],              
	},
	'ffhq_gender': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_male_train'],
		'train_bg_target_root': dataset_paths['ffhq_male_train'],
		'test_bg_source_root': dataset_paths['ffhq_male_test'],
		'test_bg_target_root': dataset_paths['ffhq_male_test'],
        
		'train_t_source_root': dataset_paths['ffhq_female_train'],
		'train_t_target_root': dataset_paths['ffhq_female_train'],
		'test_t_source_root': dataset_paths['ffhq_female_test'],
		'test_t_target_root': dataset_paths['ffhq_female_test'],              
	},
    
	'ffhq_age': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_young_train'],
		'train_bg_target_root': dataset_paths['ffhq_young_train'],
		'test_bg_source_root': dataset_paths['ffhq_young_test'],
		'test_bg_target_root': dataset_paths['ffhq_young_test'],
        
		'train_t_source_root': dataset_paths['ffhq_old_train'],
		'train_t_target_root': dataset_paths['ffhq_old_train'],
		'test_t_source_root': dataset_paths['ffhq_old_test'],
		'test_t_target_root': dataset_paths['ffhq_old_test'],            
	},
    
	'ffhq_smile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['ffhq_smile_train'],
		'train_bg_target_root': dataset_paths['ffhq_smile_train'],
		'test_bg_source_root': dataset_paths['ffhq_smile_test'],
		'test_bg_target_root': dataset_paths['ffhq_smile_test'],
        
		'train_t_source_root': dataset_paths['ffhq_nosmile_train'],
		'train_t_target_root': dataset_paths['ffhq_nosmile_train'],
		'test_t_source_root': dataset_paths['ffhq_nosmile_test'],
		'test_t_target_root': dataset_paths['ffhq_nosmile_test'],            
	},
    
	'afhqv2': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['afhqcat_train'],
		'train_bg_target_root': dataset_paths['afhqcat_train'],
		'test_bg_source_root': dataset_paths['afhqcat_test'],
		'test_bg_target_root': dataset_paths['afhqcat_test'],
        
		'train_t_source_root': dataset_paths['afhqdog_train'],
		'train_t_target_root': dataset_paths['afhqdog_train'],
		'test_t_source_root': dataset_paths['afhqdog_test'],
		'test_t_target_root': dataset_paths['afhqdog_test'], 

	},
    
	'celebaHQ_gender': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_male_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_male_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_male_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_male_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_female_train'],
		'train_t_target_root': dataset_paths['celebaHQ_female_train'],
		'test_t_source_root': dataset_paths['celebaHQ_female_test'],
		'test_t_target_root': dataset_paths['celebaHQ_female_test'],              
	},
    
	'celebaHQ_smile': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_smile_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_smile_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_smile_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_smile_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_nosmile_train'],
		'train_t_target_root': dataset_paths['celebaHQ_nosmile_train'],
		'test_t_source_root': dataset_paths['celebaHQ_nosmile_test'],
		'test_t_target_root': dataset_paths['celebaHQ_nosmile_test'],              
	},

	'celebaHQ_lipstick': {
		'transforms': transforms_config.EncodeTransforms,
		'train_bg_source_root': dataset_paths['celebaHQ_noLipstick_train'],
		'train_bg_target_root': dataset_paths['celebaHQ_noLipstick_train'],
		'test_bg_source_root': dataset_paths['celebaHQ_noLipstick_test'],
		'test_bg_target_root': dataset_paths['celebaHQ_noLipstick_test'],
        
		'train_t_source_root': dataset_paths['celebaHQ_Lipstick_train'],
		'train_t_target_root': dataset_paths['celebaHQ_Lipstick_train'],
		'test_t_source_root': dataset_paths['celebaHQ_Lipstick_test'],
		'test_t_target_root': dataset_paths['celebaHQ_Lipstick_test'],              
	}

}
