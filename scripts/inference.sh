# After launching the docker container 
EVAL_DIR='checkpoints/basemodel/checkpoints/'
CHECKPOINT='checkpoints/basemodel/checkpoints/model.bin'
VIDEO='demo/053da4e3-48ec49ba.mov'
CUDA_VISIBLE_DEVICES=5 python -m pdb src/tasks/run_adapt_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 
