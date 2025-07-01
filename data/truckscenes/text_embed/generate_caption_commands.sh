
CUDA_VISIBLE_DEVICES=0 python3 -m generate_caption  \
                                --dataset truckscenes  \
                                --caption_mode view_caption  \
                                --dataset_path "/shared/data/truckScenes/truckscenes_converted/trainval"  \
                                --output_dir "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes"  \
                                --caption_model nlpconnect/vit-gpt2-image-captioning  \
                                --tag view_caption_truckscenes

CUDA_VISIBLE_DEVICES=1 python -m generate_caption \
                                --dataset truckscenes \
                                --caption_mode scene_caption \
                                --dataset_path "/shared/data/truckScenes/truckscenes_converted/trainval" \
                                --output_dir "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes"  \
                                --image_tag ignore \
                                --view_caption_path "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes/text_embed/caption_view_truckscenes_vit-gpt2-image-captioning_view_caption_truckscenes.json"

CUDA_VISIBLE_DEVICES=2 python -m generate_caption \
                                --dataset truckscenes \
                                --caption_mode entity_caption \
                                --dataset_path "/shared/data/truckScenes/truckscenes_converted/trainval" \
                                --output_dir "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes" \
                                --image_tag ignore \
                                --view_caption_path "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes/text_embed/caption_view_truckscenes_vit-gpt2-image-captioning_view_caption_truckscenes.json" \
                                --view_caption_corr_idx_path /shared/data/truckScenes/truckscenes_converted/point2cam_correspondences

python -m generate_caption_idx_truckscenes \
        --dataset truckscenes \
        --func create_view_caption_idx \
        --cfg_file /home/daniel/spatial_understanding/benchmarks/PLA/tools/cfgs/dataset_configs/truckscenes_dataset_image.yaml \
        --view_caption_corr_idx_path /shared/data/truckScenes/truckscenes_converted/pointmap2cam_correspondences\
        --model_cfg /home/daniel/spatial_understanding/benchmarks/PLA/tools/cfgs/truckscenes_models/spconv_clip_adamw.yaml


python -m generate_caption_idx_truckscenes \
        --dataset truckscenes \
        --func create_entity_caption_idx \
        --cfg_file /home/daniel/spatial_understanding/benchmarks/PLA/tools/cfgs/dataset_configs/truckscenes_dataset_image.yaml \
        --model_cfg /home/daniel/spatial_understanding/benchmarks/PLA/tools/cfgs/truckscenes_models/spconv_clip_adamw.yaml \
        --view_caption_path "/home/daniel/spatial_understanding/benchmarks/PLA/data/truckscenes/text_embed/caption_view_truckscenes_vit-gpt2-image-captioning_view_caption_truckscenes.json" \
        --view_caption_corr_idx_path /shared/data/truckScenes/truckscenes_converted/pointmap2cam_correspondences\
        --workers 16


sh scripts/dist_train.sh 2 --cfg_file /home/daniel/spatial_understanding/benchmarks/PLA/tools/cfgs/truckscenes_models/spconv_clip_base15_caption_adamw.yaml --extra_tag exp_tag

