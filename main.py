import torch
from torch.utils.data import Dataset
import math
# import matplotlib.pyplot as plt
from utils.data_utils import NetflixPrizeDataset
import utils.data_utils
import models
from utils.logging_utils import Logger, AverageTracker
import argparse
import sys
import os
from tqdm import tqdm

# TODO:
# 8. Add gradient clipping

# Report:
# 1. Dense re-feeding, LR 0.005, tied
#       (*) main.py --batch_size 128 --momentum 0.9 --lr 0.005 --dense_refeeding_iters 1 --dataset netflix_1year --nn_arch fc1year --epochs 100
#  tmp  (*) main.py --batch_size 128 --momentum 0.9 --lr 0.001 --dense_refeeding_iters 0 --dataset netflix_3months --nn_arch fc3months --epochs 100
#  tmp2 (*) main.py --batch_size 128 --momentum 0.9 --lr 0.001 --dense_refeeding_iters 0 --dataset netflix_full --nn_arch fc_full --epochs 100
# 2. Batch-norm effect
# 3. Tie effect


def forward(dataloader, model, criterion, optimizer=None, training=True, **kwargs):
    erase_input_probability = kwargs.get("erase_input_probability", 0.0)
    dense_refeeding_iters = kwargs.get("dense_refeeding_iters", 0)
    loss_avg = AverageTracker()
    loss_clampped_avg = AverageTracker()
    acc_avg = AverageTracker()
    acc_clampped_avg = AverageTracker()
    if training:
        model.train()
    else:
        model.eval()

    for batch_idx, (batch_input, batch_output) in enumerate(tqdm(dataloader)):
        if not training:
            dense_refeeding_iters = 0
        for refeeding_iter in range(dense_refeeding_iters + 1):
            # Forward
            if refeeding_iter > 0:
                batch_input = last_output
                batch_output = last_output
            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)
            if refeeding_iter == 0:
                batch_output_mask = (batch_output != unknown_tag)
            else:
                batch_output_mask = torch.ones_like(batch_output).type_as(batch_output_mask)

            # # Hide each rating with probability of 1-bernoulli_probability
            # if erase_input_probability > 0 and training:
            #     batch_input_mask = torch.ones_like(batch_input)
            #     batch_input_bernoulli_masked = batch_input_mask.mul(1 - erase_input_probability).bernoulli()
            #     batch_input = batch_input * batch_input_bernoulli_masked.type_as(batch_input)
            # else:
            #     batch_input = batch_samples * batch_samples_mask.type_as(batch_samples)
            #     batch_output = batch_outputs * batch_outputs_mask.type_as(batch_outputs)

            # Forward
            torch.autograd.set_grad_enabled(training)
            output = model(batch_input)

            if batch_output_mask.sum().item() == 0:
                continue
            loss = criterion(output[batch_output_mask], batch_output[batch_output_mask])
            assert loss.item() == loss.item(), "Loss is NaN!!, output.shape=" + str(output.shape) + \
                                               ", batch_output_mask.shape=" + str(batch_output_mask.shape) + \
                                               ", batch_output_mask.sum().item()=" + str(batch_output_mask.sum().item()) + \
                                               ", output[batch_output_mask].shape=" + str(output[batch_output_mask].shape) + \
                                               ", batch_output[batch_output_mask].shape=" + str(batch_output[batch_output_mask].shape) + \
                                               ", output[batch_output_mask].norm()=" + str(output[batch_output_mask]) + \
                                               ", WeightsNorm=" + str(sum([p.norm() for p in model.parameters()])) + \
                                               ", refeeding_iter=" + str(refeeding_iter)
            loss_avg.add(loss.item(), batch_output.size(0))
            if not training:
                # Calculate clampped loss (using prior that ratings are in interval [1,5])
                loss_clampped = criterion(output[batch_output_mask].clamp(min=1, max=5),
                                          batch_output[batch_output_mask])
                assert loss_clampped.item() == loss_clampped.item(), "Loss is NaN!!"
                loss_clampped_avg.add(loss_clampped.item(), batch_output.size(0))

            torch.autograd.set_grad_enabled(False)

            if training:
                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Optimizer step
                optimizer.step()

            # Dense re-feeding
            if training and dense_refeeding_iters > 0:
                last_output = output.detach()

            # Calculate how many tags were predicted correctly
            num_of_correct = (torch.round(output[batch_output_mask]) == torch.round(
                batch_output[batch_output_mask])).sum().item()

            num_of_items = batch_output_mask.sum().item()
            if num_of_items > 0:
                acc_avg.add((num_of_correct / num_of_items), batch_input.size(0))

            # Calculate how many tags were predicted correctly (with clampping)
            num_of_correct = (torch.round(output[batch_output_mask].clamp(min=1, max=5)) == torch.round(
                batch_output[batch_output_mask])).sum().item()

            num_of_items = batch_output_mask.sum().item()
            if num_of_items > 0:
                acc_clampped_avg.add((num_of_correct / num_of_items), batch_input.size(0))
    return acc_avg.avg, math.sqrt(loss_avg.avg), acc_clampped_avg.avg, math.sqrt(loss_clampped_avg.avg)


###########################################################################
# Script's arguments
###########################################################################
run_cmd = ""
for arg in sys.argv:
    run_cmd += arg + " "

datasets_names = sorted(name for name in utils.data_utils.__dict__
                        if name.islower() and not name.startswith("_")
                        and callable(utils.data_utils.__dict__[name]))

archs_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Train and record statistics of a Neural Network')
parser.add_argument('--logname', type=str, required=True,
                    help='Prefix of logfile name')
parser.add_argument('--results_dir', type=str, default="TMP",
                    help='Results dir name')

parser.add_argument('--dataset', default="netflix_3months", type=str, choices=datasets_names,
                    help='The name of the dataset to train and test on. [Default: netflix_3months]')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size')

parser.add_argument('--nn_arch', type=str, required=True, choices=archs_names,
                    help='Neural network architecture')


parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate initial value')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum initial value')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay initial value')
parser.add_argument('--epochs', default=5, type=int,
                    help='Number of epochs')
parser.add_argument('--desc', default="", type=str, nargs='*',
                    help='Desc file content')

parser.add_argument('--dense_refeeding_iters', default=0, type=int,
                    help='Number of dense re-feedings')

parser.add_argument('--erase_input_probability', default=0.0, type=float,
                    help='Probability of erasing inputs elements during training')



args = parser.parse_args()

torch.autograd.set_grad_enabled(False)

###########################################################################
# Logging
###########################################################################
save_path = os.path.join("./logs", str(args.results_dir) + "/")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create logger
logger = Logger(True, save_path + args.logname, True, True)
logger.info(run_cmd)

logger.info("Script args: " + str(args))

if args.desc != "":
    logger.create_desc_file(" ".join(args.desc))
else:
    logger.create_desc_file(str(args))

# Add logfile name to last runs

lastlogs_logger = Logger(add_timestamp=False, logfile_name="last_logs.txt", logfile_name_time_suffix=False,
                         print_to_screen=False)
lastlogs_logger.info(logger.get_log_basename() + " ")
lastlogs_logger = None

###########################################################################
# cuda
###########################################################################
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

###########################################################################
# Data
###########################################################################
# Create train data loader
ratings_file = "./netflix-prize-data/all_data.txt"
# ratings_file = "./netflix-prize-data/combined_data_1.txt"
parsed_file = "./netflix-prize-data/processed_data.pth"
train_range = {"start": {"year": 2005, "month": 9},
               "end": {"year": 2005, "month": 11}}
test_range = {"start": {"year": 2005, "month": 12},
              "end": {"year": 2005, "month": 12}}


netflix_prize_dataset_train, netflix_prize_dataset_test = utils.data_utils.__dict__[args.dataset]()

train_dataloader = torch.utils.data.DataLoader(netflix_prize_dataset_train, shuffle=True, batch_size=args.batch_size)
test_dataloader = torch.utils.data.DataLoader(netflix_prize_dataset_test, shuffle=False, batch_size=args.batch_size)

# Create model
model_type = models.__dict__[args.nn_arch]
model = model_type(input_size=netflix_prize_dataset_train.num_of_movies,
                   input_dropout_coef=args.erase_input_probability)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

# Print number of parameters
num_of_params = 0
for parameter in model.parameters():
    num_of_params += parameter.numel()
logger.info("Model has " + str(num_of_params) + " parameters")

# Configuration
num_of_epochs = args.epochs
criterion = torch.nn.MSELoss()
sgd_params = {
    "momentum": args.momentum,
    "lr": args.lr,
    "weight_decay": args.weight_decay
}
optimizer = torch.optim.SGD(model.parameters(), **sgd_params)
sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)


# The rating of a non-rated movie
unknown_tag = 0

train_epc_loss = []
train_epc_acc = []
test_epc_loss = []
test_epc_acc = []
test_epc_loss_clampped = []
test_epc_acc_clampped = []

for epoch in range(num_of_epochs):
    sched.step()
    avg_acc, avg_loss, _, _ = forward(dataloader=train_dataloader, model=model, criterion=criterion,
                                      optimizer=optimizer, training=True,
                                      dense_refeeding_iters=args.dense_refeeding_iters,
                                      erase_input_probability=args.erase_input_probability)
    train_epc_loss.append(avg_loss)
    train_epc_acc.append(avg_acc)
    logger.info("[Train] Epoch=" + str(epoch) + ", Avg Loss is " + str(avg_loss) + ", Avg acc is " + str(avg_acc))
    avg_acc, avg_loss, avg_clampped_acc, avg_clampped_loss = forward(dataloader=test_dataloader, model=model,
                                                                     criterion=criterion, optimizer=None,
                                                                     training=False)
    test_epc_loss.append(avg_loss)
    test_epc_acc.append(avg_acc)
    test_epc_loss_clampped.append(avg_clampped_loss)
    test_epc_acc_clampped.append(avg_clampped_acc)

    logger.info("[Test] Epoch=" + str(epoch) + ", Avg Loss is " + str(avg_loss) + " (" + str(avg_clampped_loss) +
                "), Avg acc is " + str(avg_acc))

logger.save_variables({"train_iter_loss": None,
                       "train_iter_acc": None,
                       "train_epc_loss": train_epc_loss,
                       "train_epc_acc": train_epc_acc,
                       "test_epc_loss": test_epc_loss,
                       "test_epc_acc": test_epc_acc,
                       "test_epc_loss_clampped": test_epc_loss_clampped,
                       "test_epc_acc_clampped": test_epc_acc_clampped
                       }, var_name="loss_acc")
