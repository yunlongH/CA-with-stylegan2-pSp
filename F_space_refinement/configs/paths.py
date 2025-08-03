from dataclasses import dataclass, field, fields
import os

models_dir = "pretrained_models/"

# 1) define overrides for *any* field you want to change per‐dataset

specific_paths = {
    "ffhq_glasses": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "e4e_path": models_dir + "e4e_ffhq_encode.pt",
        "e4e_cs_path": models_dir + "100000_iter_e4e_cs.pt",
        "pSp_cs_path": models_dir + "pSp_cs_149900_iter.pt"
        #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_glasses_sx/checkpoints/iteration_140000.pt"
        #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_cs1s2/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_140000.pt"
    },
    "ffhq_glasses_sx": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_glasses_sx/checkpoints/iteration_130000.pt"
    },
    "ffhq_gender_sx": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_sx/checkpoints/iteration_140000.pt",

    },


    "afhq_cat_dog": {
        "stylegan_weights":     "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt",
        "stylegan_weights_pkl": "/home/ids/yuhe/Projects/CA_with_GAN/3_code/styleGAN/pretrained_models/stylegan2/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pkl",
        "psp_path":             "../../pretrained_models/pSp_models/psp_afhqv2.pt",
        #"pSp_cs_path":       "../../pretrained_models/pSp_cs_models/pSp_cs_dogcat_baseline.pt",
        #"pSp_cs_path":       "../../pretrained_models/pSp_cs_models/pSp_cs_dogcat_sx.pt", # pSp_cs_dogcat_baseline.pt
        "pSp_cs_path":       "../../pSp_CS-StyleGAN/results/AFHQ/dog_cat_c_s1_s2/checkpoints/iteration_100000.pt",
        # "e4e_cs_path":          models_dir + "100000_iter_e4e_cs_afhq.pt"
    },

    "brats_edit": {
        "stylegan_weights":     "../../pretrained_models/pSp_models/stylegan2-brats_880k.pt",
        "stylegan_weights_pkl": "../../pretrained_models/pSp_models/stylegan2-brats_880k.pt",
        #  "psp_path": "../../pretrained_models/pSp_models/brats/psp_brats_160k.pt",
        # "pSp_cs_path": "../../pretrained_models/pSp_cs_models/brats/12layers_resume_30k.pt",
       "psp_path": "../../pretrained_models/pSp_models/brats/psp_brats_220k.pt",
       "pSp_cs_path": "../../pretrained_models/pSp_cs_models/brats/iteration_150000.pt",
    },

    "ffhq_gender": {
        "stylegan_weights":    "../../pretrained_models/pSp_models/stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": "../../pretrained_models/pSp_models/stylegan2-ffhq-config-f.pkl",
        "psp_path": "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender/checkpoints/iteration_140000.pt",
        #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_gender_sx/checkpoints/iteration_320000.pt"
    },

    "ffhq_age": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/other_attributes/ffhq_age/12layers_lr0.01/checkpoints/iteration_200000.pt"
    },
    "ffhq_pose": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/other_attributes/ffhq_pose/12layers_lr0.01/checkpoints/iteration_210000.pt"
    },
    "ffhq_smile": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/ffhq_smile_v2/checkpoints/iteration_200000.pt"
    },

    "celebahq_smile": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ_smile/checkpoints/iteration_120000.pt"
    },
    "celebahq_gender": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_celebahq_styleganffhq.pt",
        #"psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ/celebaHQ_gender/checkpoints/iteration_120000.pt",
        #"pSp_cs_path": "../../pSp_CS-StyleGAN/results/celebaHQ/celebaHQ_ffhq_gender/checkpoints/iteration_90000.pt"
    },
    "ffhq_glasses_smile": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/glasses_smile/lr0.01/checkpoints/iteration_220000.pt"
    },
    "ffhq_glassesvssmile": {
        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "psp_path":  "../../pretrained_models/pSp_models/psp_ffhq_encode.pt",
        "pSp_cs_path": "../../pSp_CS-StyleGAN/results/glassesVSsmile/10layers_lr0.01_s1s2_id0.4/checkpoints/iteration_40000.pt"
    },
    "afhq": {
        "stylegan_weights":     "../../pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pt",
        "stylegan_weights_pkl": "../../pretrained_models/stylegan2_NGC_catalog/stylegan2-afhqv2-512x512.pkl",
    },
    "lsun_church": {

        "stylegan_weights":     models_dir + "stylegan2-ffhq-config-f.pt",
        "stylegan_weights_pkl": models_dir + "stylegan2-ffhq-config-f.pkl",
        "e4e_path":  "../../pretrained_models/e4e_models/e4e_church_encode.pt",
        "e4e_cs_path": "../e4e/results/12nmlp_lr0.01_church_resume/checkpoints/iteration_24000.pt",
       # "e4e_cs_path": "../e4e/results/12nmlp_lr0.001_church/checkpoints/iteration_160000.pt"
    },

}
        # if self.opts.e4e_checkpoint_path is not None:
        #     print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.e4e_checkpoint_path))
        #     ckpt = torch.load(self.opts.e4e_checkpoint_path, map_location='cpu', weights_only=True)
        #     self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        #     self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
# SOTA_encoders_StyleGAN/e4e/results/12nmlp_lr0.01_church_resume/checkpoints/iteration_24000.pt

@dataclass
class DefaultPathsClass:
    dataset: str = os.getenv("DATASET", "ffhq_glasses")

    # Base paths
    farl_path:      str = models_dir + "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    mobile_net_pth: str = models_dir + "mobilenet0.25_Final.pth"
    ir_se50_path:   str = models_dir + "model_ir_se50.pth"
    stylegan_car_weights: str = models_dir + "stylegan2-car-config-f-new.pkl"
    arcface_model_path:   str = models_dir + "iresnet50-7f187506.pth"
    moco:                 str = models_dir + "moco_v2_800ep_pretrain.pt"
    curricular_face_path: str = models_dir + "CurricularFace_Backbone.pth"
    mtcnn:                str = models_dir + "mtcnn"
    landmark:             str = models_dir + "79999_iter.pth"

    # Optional — can be filled by dataset-specific config
    stylegan_weights:     str = None
    stylegan_weights_pkl: str = None
    psp_path:             str = None
    e4e_path:             str = None
    e4e_cs_path:          str = None
    pSp_cs_path:          str = None

    def __post_init__(self):
        key = self.dataset.lower()
        overrides = specific_paths.get(key, {})

        if not overrides:
            raise ValueError(f"No specific path config found for dataset '{self.dataset}'.")

        for name, val in overrides.items():
            # Append if not already set or explicitly None
            if not hasattr(self, name) or getattr(self, name) is None:
                setattr(self, name, val)
            else:
                print(f"[INFO] Keeping existing value for '{name}', not overridden.")

        # Sanity check for critical fields
        if self.stylegan_weights is None or self.stylegan_weights_pkl is None:
            raise ValueError(f"Missing required stylegan weights for dataset '{self.dataset}'.")

    def __iter__(self):
        for f in fields(self):
            yield f.name, getattr(self, f.name)

# Singleton instance
DefaultPaths = DefaultPathsClass()
