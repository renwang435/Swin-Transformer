# Colorization

For pretraining Swin backbone for colorization (i.e. grayscale images --> ImageNet classification)

## Environment


```bash
conda env create -f swin.yaml
conda activate swin
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

```


## Training

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
  --cfg configs/swin/colorization_swin_small_patch4_window7_224.yaml \
  --data-path /path/to/ILSVRC2012 \
  --batch-size 512 --output output/colorization_bs_512 \
  --opts MODEL.COLORIZATION True DATA.NUM_WORKERS 8 DATA.PIN_MEMORY True
```
