import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data", help = "data")
parser.add_argument("density", help="density")
parser.add_argument("alpha_meta", type=float, help="learning rate of metagraph")
parser.add_argument("regul_meta", type=float, help="regularization rate of metagraph")
parser.add_argument("alpha", type=float, help="learning rate of original graph")
parser.add_argument("regul", type=float, help="regularization rate of original graph")
args = parser.parse_args()

data = args.data
density = args.density
alpha_meta = args.alpha_meta
regul_meta = args.regul_meta
alpha = args.alpha
regul_rate = args.regul

path_meta = './result_square/' + density + '/' + data + '_meta/'
path_data = "./benchmarks/" + data + '/'
path_meta=path_data

train_dataloader_meta = TrainDataLoader(
	in_path = path_meta, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

test_dataloader_meta = TestDataLoader(path_meta, "link", False)

distmult_meta = DistMult(
	ent_tot = train_dataloader_meta.get_ent_tot(),
	rel_tot = train_dataloader_meta.get_rel_tot(),
	dim = 200
)

model_meta = NegativeSampling(
	model = distmult_meta, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader_meta.get_batch_size(), 
	regul_rate = regul_meta
)

trainer_meta = Trainer(model = model_meta, data_loader = train_dataloader_meta, train_times = 500, alpha = alpha_meta, use_gpu = True, opt_method = "adagrad")
trainer_meta.run()


# save imbedding
import pickle
entity_embeddings = distmult_meta.ent_embeddings.weight.data.cpu().numpy()
relation_embeddings = distmult_meta.rel_embeddings.weight.data.cpu().numpy()

SAVE_PATH = './embedding/'+ data + '/'
#print(relation_embeddings)
f1 = open(SAVE_PATH+'entity_embeddings_origin', 'wb')
f2 = open(SAVE_PATH+'relation_embeddings_origin', 'wb')
pickle.dump(entity_embeddings, f1)
pickle.dump(relation_embeddings, f2)
f1.close()
f2.close()

tester = Tester(model = distmult_meta, data_loader = test_dataloader_meta, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
