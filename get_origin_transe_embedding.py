import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help="data")
parser.add_argument("density", help="density")
parser.add_argument("alpha_meta", type=float, help="learning rate of metagraph")
parser.add_argument("margin_meta", type=float, help="margin of metagraph")
parser.add_argument("alpha", type=float, help="learning rate of original graph")
parser.add_argument("margin", type=float, help="margin of original graph")
args = parser.parse_args()

data = args.data
density = args.density
alpha_meta = args.alpha_meta
margin_meta = args.margin_meta
alpha = args.alpha
margin = args.margin

path_meta = './result_square/' + density + '/' + data + '_meta/'
path_origin = "./benchmarks/" + data + '/'
print(0)
train_dataloader_origin = TrainDataLoader(
	in_path = path_origin, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

test_dataloader_origin = TestDataLoader(path_origin, "link", False)
print(1)
transe_origin = TransE(
	ent_tot = train_dataloader_origin.get_ent_tot(),
	rel_tot = train_dataloader_origin.get_rel_tot(),
	dim = 100,
	p_norm = 1,
	norm_flag = True
)
print(2)
model_origin = NegativeSampling(
	model = transe_origin, 
	loss = MarginLoss(margin = margin_meta),
	batch_size = train_dataloader_origin.get_batch_size()
)

trainer_origin = Trainer(model = model_origin, data_loader = train_dataloader_origin, train_times = 200, alpha = alpha_meta, use_gpu = True)
trainer_origin.run()

# save imbedding
import pickle
entity_embeddings = transe_origin.ent_embeddings.weight.data.cpu().numpy()
relation_embeddings = transe_origin.rel_embeddings.weight.data.cpu().numpy()

SAVE_PATH = './embedding/'+ data + '/'
print(relation_embeddings)
f1 = open(SAVE_PATH+'entity_embeddings_origin', 'wb')
f2 = open(SAVE_PATH+'relation_embeddings_origin', 'wb')
pickle.dump(entity_embeddings, f1)
pickle.dump(relation_embeddings, f2)
f1.close()
f2.close()

tester = Tester(model = transe_origin, data_loader = test_dataloader_origin, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
