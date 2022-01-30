import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "data")
parser.add_argument("density", help="density")
parser.add_argument("alpha_meta", type=float, help="learning rate of metagraph")
parser.add_argument("margin_meta", type=float, help="margin of metagraph")
parser.add_argument("adv_meta", type=float, help="adv rate of metagraph")
parser.add_argument("alpha", type=float, help="learning rate of original graph")
parser.add_argument("margin", type=float, help="margin of original graph")
parser.add_argument("adv", type=float, help="adv of original graph")
args = parser.parse_args()

data = args.data
density = args.density
alpha_meta = args.alpha_meta
margin_meta = args.margin_meta
adv_meta = args.adv_meta
alpha = args.alpha
margin = args.margin
adv_temperature = args.adv

path_meta = './result_square/' + density + '/' + data + '_meta/'
path_data = "./benchmarks/" + data + '/'
path_meta=path_data
train_dataloader = TrainDataLoader(
	in_path = path_meta, 
	batch_size = 1000,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

test_dataloader = TestDataLoader(path_meta, "link", False)

rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 512,
	margin = margin_meta,
	epsilon = 2,
)

model_meta = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = adv_meta),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

trainer = Trainer(model = model_meta, data_loader = train_dataloader, train_times = 500, alpha = alpha_meta, use_gpu = True, opt_method = "adam")
trainer.run()

# save imbedding
import pickle
entity_embeddings = rotate.ent_embeddings.weight.data.cpu().numpy()
relation_embeddings = rotate.rel_embeddings.weight.data.cpu().numpy()

SAVE_PATH = './embedding/'+ data + '/'
#print(relation_embeddings)
f1 = open(SAVE_PATH+'entity_embeddings_origin', 'wb')
f2 = open(SAVE_PATH+'relation_embeddings_origin', 'wb')
pickle.dump(entity_embeddings, f1)
pickle.dump(relation_embeddings, f2)
f1.close()
f2.close()

tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)