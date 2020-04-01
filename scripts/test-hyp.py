#!/usr/bin/env python3

# TTA
# Remove Images
# Correct Normailzation
# Model split correctly

import argparse
from tqdm import tqdm
from fastai2.vision.all import *
from efficientnet_pytorch import EfficientNet
from manifold_mixup import *
from dynamic_mixup import *
import telegram

def bot_msg(msg):
    bot = telegram.Bot(token='827844759:AAHMY20tSCOllZFjaXqv0Jy-YU7q1w08v6U')
    bot.send_message(chat_id=-373784678, text=msg)

MODEL =  EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)
exp_time = str(datetime.now().replace(second=0, microsecond=0)).replace(" ", "_")
SIZE = 512
TEST_PATH = "/home/ubuntu/crop/data_v1/test"

parser = argparse.ArgumentParser(description="Best Method")
parser.add_argument('--gpu', type=int, default=3, help="GPU ")
parser.add_argument('--repeat', type=int, default=20, help="Repeat")
parser.add_argument('--size', type=int, default=512, help="Size")
parser.add_argument('--bs', type=int, default=8, help="Batch Size")
parser.add_argument('--exp', type=int, default=1, help="Exp")
args = parser.parse_args()
print(args)

EXP_NAME = "{}-{}-{}".format(args.bs, args.size, exp_time)
print(EXP_NAME)

ACCURACY_ALL = []

if args.exp == 1:
    item_tfms = [Resize(args.size), Rotate(), Flip()]

cbs = [DynamicOutputMixup(scheduler=SchedLin, alpha_max=0., alpha_min=1.)]
cbs_1 = EarlyStoppingCallback(patience=3, min_delta=1e-5, monitor="error_rate")
cbs_2 = SaveModelCallback(fname=EXP_NAME)

for i in tqdm(range(args.repeat)):
    import gc; gc.collect()
    i = i%5+1
    CSV_PATH = "/home/ubuntu/crop/notebooks/fastai2/Prod/cv{}.csv".format(i)
    df = pd.read_csv(CSV_PATH)

    dls = ImageDataLoaders.from_df(df=df, path="/home/ubuntu/crop/data_v3/train", item_tfms=item_tfms,
        bs=args.bs, num_workers=8, device=args.gpu)

    opt_func = partial(ranger)
    learn = Learner(dls, MODEL, metrics=[error_rate, accuracy],
        opt_func=opt_func, cbs=cbs).to_fp16()

    learn.freeze()
    learn.fit_one_cycle(100, slice(2e-3), cbs=[cbs_1, cbs_2])

    learn.unfreeze()
    learn.fit_one_cycle(100, slice(1e-7, 1e-4), cbs=[cbs_1, cbs_2])

    ac_1 = learn.validate()
    print(ac_1[-1])
    ACCURACY_ALL.append(ac_1[-1])

print(ACCURACY_ALL)
print("Average Accuracy {0:.5f}".format(np.mean(ACCURACY_ALL)))

bot_msg("#########")
bot_msg(str(args))
bot_msg(ACCURACY_ALL)
bot_msg(np.mean(ACCURACY_ALL))
bot_msg("#########")

