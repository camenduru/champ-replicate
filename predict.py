import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/champ')
os.chdir('/content/champ')

import os.path as osp
from datetime import datetime
from pathlib import Path as MyPath

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.guidance_encoder import GuidanceEncoder
from models.champ_model import ChampModel

from pipelines.pipeline_aggregation import MultiGuidance2LongVideoPipeline

from utils.video_utils import resize_tensor_frames, save_videos_grid, pil_list_to_tensor

class ExperimentConfig:
    def __init__(self, exp_name, width, height, data, seed, base_model_path, vae_model_path,
                 image_encoder_path, ckpt_dir, motion_module_path, num_inference_steps,
                 guidance_scale, enable_zero_snr, weight_dtype, guidance_types,
                 noise_scheduler_kwargs, unet_additional_kwargs, guidance_encoder_kwargs,
                 enable_xformers_memory_efficient_attention):
        self.exp_name = exp_name
        self.width = width
        self.height = height
        self.data = data
        self.seed = seed
        self.base_model_path = base_model_path
        self.vae_model_path = vae_model_path
        self.image_encoder_path = image_encoder_path
        self.ckpt_dir = ckpt_dir
        self.motion_module_path = motion_module_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.enable_zero_snr = enable_zero_snr
        self.weight_dtype = weight_dtype
        self.guidance_types = guidance_types
        self.noise_scheduler_kwargs = noise_scheduler_kwargs
        self.unet_additional_kwargs = unet_additional_kwargs
        self.guidance_encoder_kwargs = guidance_encoder_kwargs
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    for guidance_type in cfg.guidance_types:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs['guidance_embedding_channels'],
            guidance_input_channels=cfg.guidance_encoder_kwargs['guidance_input_channels'],
            block_out_channels=cfg.guidance_encoder_kwargs['block_out_channels'],
        ).to(device="cuda", dtype=weight_dtype)
    
    return guidance_encoder_group

def process_semantic_map(semantic_map_path: MyPath):
    image_name = semantic_map_path.name
    mask_path = semantic_map_path.parent.parent / "mask" / image_name
    semantic_array = np.array(Image.open(semantic_map_path))
    mask_array = np.array(Image.open(mask_path).convert("RGB"))
    semantic_pil = Image.fromarray(np.where(mask_array > 0, semantic_array, 0))
    
    return semantic_pil

def combine_guidance_data(cfg):
    guidance_types = cfg.guidance_types
    guidance_data_folder = cfg.data['guidance_data_folder']
    
    guidance_pil_group = dict()
    for guidance_type in guidance_types:
        guidance_pil_group[guidance_type] = []
        for guidance_image_path in sorted(MyPath(osp.join(guidance_data_folder, guidance_type)).iterdir()):
            # Add black background to semantic map
            if guidance_type == "semantic_map":
                guidance_pil_group[guidance_type] += [process_semantic_map(guidance_image_path)]
            else:
                guidance_pil_group[guidance_type] += [Image.open(guidance_image_path).convert("RGB")]
    
    # get video length from the first guidance sequence
    first_guidance_length = len(list(guidance_pil_group.values())[0])
    # ensure all guidance sequences are of equal length
    assert all(len(sublist) == first_guidance_length for sublist in list(guidance_pil_group.values()))
    
    return guidance_pil_group, first_guidance_length

def inference(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    ref_image_pil,
    guidance_pil_group,
    video_length,
    width,
    height,
    device="cuda",
    dtype=torch.float16,
):
    reference_unet = model.reference_unet
    denoising_unet = model.denoising_unet
    guidance_types = cfg.guidance_types
    guidance_encoder_group = {f"guidance_encoder_{g}": getattr(model, f"guidance_encoder_{g}") for g in guidance_types}
    
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    pipeline = MultiGuidance2LongVideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype)
    
    video = pipeline(
        ref_image_pil,
        guidance_pil_group,
        width,
        height,
        video_length,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator
    ).videos
    
    del pipeline
    torch.cuda.empty_cache()
    
    return video

class Predictor(BasePredictor):
    def setup(self) -> None:

        # Define the YAML configuration data
        yaml_data = {
            "exp_name": "Animation",
            "width": 512,
            "height": 512,
            "data": {
                "ref_image_path": 'example_data/ref_images/ref-01.png',
                "guidance_data_folder": 'example_data/motions/motion-01',
            },
            "seed": 42,
            "base_model_path": 'pretrained_models/stable-diffusion-v1-5',
            "vae_model_path": 'pretrained_models/sd-vae-ft-mse',
            "image_encoder_path": 'pretrained_models/image_encoder',
            "ckpt_dir": 'pretrained_models/champ',
            "motion_module_path": 'pretrained_models/champ/motion_module.pth',
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "enable_zero_snr": True,
            "weight_dtype": "fp16",
            "guidance_types": ['depth', 'normal', 'semantic_map', 'dwpose'],
            "noise_scheduler_kwargs": {
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "linear",
                "steps_offset": 1,
                "clip_sample": False,
            },
            "unet_additional_kwargs": {
                "use_inflated_groupnorm": True,
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": False,
                "use_motion_module": True,
                "motion_module_resolutions": [1, 2, 4, 8],
                "motion_module_mid_block": True,
                "motion_module_decoder_only": False,
                "motion_module_type": "Vanilla",
                "motion_module_kwargs": {
                    "num_attention_heads": 8,
                    "num_transformer_block": 1,
                    "attention_block_types": ['Temporal_Self', 'Temporal_Self'],
                    "temporal_position_encoding": True,
                    "temporal_position_encoding_max_len": 32,
                    "temporal_attention_dim_div": 1,
                },
            },
            "guidance_encoder_kwargs": {
                "guidance_embedding_channels": 320,
                "guidance_input_channels": 3,
                "block_out_channels": [16, 32, 96, 256],
            },
            "enable_xformers_memory_efficient_attention": True,
        }

        # Create an instance of ExperimentConfig
        self.cfg = ExperimentConfig(**yaml_data)

        # setup pretrained models
        if self.cfg.weight_dtype == "fp16":
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32
            
        sched_kwargs = self.cfg.noise_scheduler_kwargs
        if self.cfg.enable_zero_snr:
            sched_kwargs.update( 
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        self.noise_scheduler = DDIMScheduler(**sched_kwargs)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})

        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            self.cfg.image_encoder_path,
        ).to(dtype=self.weight_dtype, device="cuda")

        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_path).to(
            dtype=self.weight_dtype, device="cuda"
        )

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.cfg.base_model_path,
            self.cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=self.cfg.unet_additional_kwargs,
        ).to(dtype=self.weight_dtype, device="cuda")

        reference_unet = UNet2DConditionModel.from_pretrained(
            self.cfg.base_model_path,
            subfolder="unet",
        ).to(device="cuda", dtype=self.weight_dtype)

        guidance_encoder_group = setup_guidance_encoder(self.cfg)

        ckpt_dir = self.cfg.ckpt_dir
        denoising_unet.load_state_dict(
            torch.load(
                osp.join(ckpt_dir, f"denoising_unet.pth"),
                map_location="cpu",
            ),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(
                osp.join(ckpt_dir, f"reference_unet.pth"),
                map_location="cpu",
            ),
            strict=False,
        )

        for guidance_type, guidance_encoder_module in guidance_encoder_group.items():
            guidance_encoder_module.load_state_dict(
                torch.load(
                    osp.join(ckpt_dir, f"guidance_encoder_{guidance_type}.pth"),
                    map_location="cpu",
                ),
                strict=False,
            )
            
        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )
            
        self.model = ChampModel(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            reference_control_writer=reference_control_writer,
            reference_control_reader=reference_control_reader,
            guidance_encoder_group=guidance_encoder_group,
        ).to("cuda", dtype=self.weight_dtype)

        if self.cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                reference_unet.enable_xformers_memory_efficient_attention()
                denoising_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
    def predict(
        self,
        ref_image_path: Path = Input(description="Image"),
        guidance_data: str = Input(choices=["example_data/motions/motion-01",
                                            "example_data/motions/motion-02",
                                            "example_data/motions/motion-03",
                                            "example_data/motions/motion-04",
                                            "example_data/motions/motion-05",
                                            "example_data/motions/motion-06",
                                            "example_data/motions/motion-07",
                                            "example_data/motions/motion-08",
                                            "example_data/motions/motion-09"
                                            ], default="example_data/motions/motion-01"),
    ) -> Path:
        cfg.data['guidance_data_folder'] = guidance_data
        ref_image_pil = Image.open(ref_image_path)
        ref_image_w, ref_image_h = ref_image_pil.size
        guidance_pil_group, video_length = combine_guidance_data(self.cfg)
        result_video_tensor = inference(
            cfg=self.cfg,
            vae=self.vae,
            image_enc=self.image_enc,
            model=self.model,
            scheduler=self.noise_scheduler,
            ref_image_pil=ref_image_pil,
            guidance_pil_group=guidance_pil_group,
            video_length=video_length,
            width=self.cfg.width, height=self.cfg.height,
            device="cuda", dtype=self.weight_dtype
        )  # (1, c, f, h, w)
        result_video_tensor = resize_tensor_frames(result_video_tensor, (ref_image_h, ref_image_w))
        save_videos_grid(result_video_tensor, osp.join("/content", "animation.mp4"))
        return Path('/content/animation.mp4')