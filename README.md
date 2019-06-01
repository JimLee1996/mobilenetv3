# A PyTorch implementation of MobileNetV3

This is a PyTorch implementation of MobileNetV3 in the paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

We have tried to follow every detail in the original paper. Discussions are welcomed to improve this work.

**[News]** We are training MobileNetV3 Small on ImageNet.

## Details

See implementation in [mobilenetv3.py](mobilenetv3.py) and training code in [main.py](main.py).

## MobileNetV3 large

|              | Madds     | #Params | Top1-acc  | Pretrained Model |
| -----------  | --------- | ---------- | --------- | ---------------- |
| Offical  | 219 M     | 5.4  M     | 75.2%     | - |
| Ours     |   -       | 5.11 M     |  -        | - |

## MobileNetV3 small
|              | Madds     | #Params | Top1-acc  | Pretrained Model |
| -----------  | --------- | ------- | --------- | ---------------- |
| Offical  | 66 M      | 2.9  M     | 67.4%  | - |
| Ours     |   -       | 3.11 M     | -      | - |

## Usage

```
➜  python main.py --help
usage: main.py [-h] [--data-path DATA_PATH] [--device DEVICE] [-b BATCH_SIZE]
               [--epochs N] [-j N] [--lr LR] [--momentum M] [--wd W]
               [--lr-step-size LR_STEP_SIZE] [--lr-gamma LR_GAMMA]
               [--print-freq PRINT_FREQ] [--output-dir OUTPUT_DIR]
               [--resume RESUME] [--start-epoch N] [--cache-dataset]
               [--sync-bn] [--test-only] [--pretrained]
               [--world-size WORLD_SIZE] [--dist-url DIST_URL]

PyTorch Classification Training

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        dataset
  --device DEVICE       device
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 16)
  --lr LR               initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --lr-step-size LR_STEP_SIZE
                        decrease lr every step-size epochs
  --lr-gamma LR_GAMMA   decrease lr by a factor of lr-gamma
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path where to save model
  --resume RESUME       resume from checkpoint
  --start-epoch N       start epoch
  --cache-dataset       Cache the datasets for quicker initialization. It also
                        serializes the transforms
  --sync-bn             Use sync batch norm
  --test-only           Only test the model
  --pretrained          Use pre-trained models from the modelzoo
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
```
