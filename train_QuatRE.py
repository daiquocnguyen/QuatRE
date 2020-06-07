from Config import *
from QuatRE import *
import json
import os


from argparse import ArgumentParser
parser = ArgumentParser("QuatRE")
parser.add_argument("--dataset", default="WN18RR", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate")
parser.add_argument("--nbatches", default=100, type=int, help="Number of batches")
parser.add_argument("--num_epochs", default=8000, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='QuatRE', help="")
parser.add_argument('--save_steps', type=int, default=10000, help='')
parser.add_argument('--neg_num', default=1, type=int, help='')
parser.add_argument('--hidden_size', type=int, default=4, help='')
parser.add_argument('--valid_steps', type=int, default=400, help='')
parser.add_argument("--lmbda", default=0.1, type=float, help="")
parser.add_argument("--lmbda2", default=0.01, type=float, help="")
parser.add_argument("--optim", default='adagrad', help="")
args = parser.parse_args()

print(args)
out_dir = os.path.abspath(os.path.join("../runs_QuatRE/"))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
result_dir = os.path.abspath(os.path.join(checkpoint_dir, args.model_name))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

con = Config()
con.set_in_path("./benchmarks/" + args.dataset + "/")
con.set_work_threads(8)
con.set_train_times(args.num_epochs)
con.set_nbatches(args.nbatches)
con.set_alpha(args.learning_rate)
con.set_bern(1)
con.set_dimension(args.hidden_size)
con.set_lmbda(args.lmbda)
con.set_margin(1.0)
con.set_ent_neg_rate(args.neg_num)
con.set_opt_method(args.optim)
con.set_save_steps(args.save_steps)
con.set_valid_steps(args.valid_steps)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(checkpoint_dir)
con.set_result_dir(result_dir)
con.set_test_link(True)
# con.set_test_triple(True)
con.init()
con.set_train_model(QuatRE)
con.training_model()
# con.training_triple_classification()
