import argparse
import os
import sys

import torch
from autogllight.nas.algorithm import Graces
from autogllight.nas.estimator import OneShotOGBEstimator
from autogllight.nas.space import GracesSpace
from autogllight.utils import set_seed
from autogllight.utils.evaluation import Auc

import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

sys.path.append("..")
sys.path.append(".")
os.environ["AUTOGL_BACKEND"] = "pyg"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DegreeDistribution(object):
    def __call__(self, g):
        '''if g.is_undirected():
            edges = g.edge_index[0]
        else:
            edges = torch.cat((g.edge_index[0], g.edge_index[1]))'''
        edges = g.edge_index[1]
        if edges.numel() == 0:
            deratio = torch.tensor([0.0, 0.0, 0.0])
        else:
            degrees = torch_geometric.utils.degree(edges).to(torch.long).numpy().tolist()
            deratio = [degrees.count(i) for i in range(1, 4)]
            deratio = torch.tensor(deratio) / g.num_nodes
        g.deratio = deratio
        return g

def load_data(dataset_name='', batch_size=32):
    transform = DegreeDistribution()
    if 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(name=dataset_name, root='./ogb/dataset/', pre_transform=transform)
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        return [dataset, dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], train_loader, val_loader, test_loader], -1
    
def parser_args():
    parser = argparse.ArgumentParser("pas-train-search")
    parser.add_argument(
        "--data", type=str, default="ogbg-molbace", help="location of the data corpus"
    )
    parser.add_argument(
        "--record_time",
        action="store_true",
        default=False,
        help="used for run_with_record_time func",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="init learning rate"
    )
    parser.add_argument(
        "--learning_rate_min", type=float, default=0.001, help="min learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--gpu", type=int, default=3, help="gpu device id")
    parser.add_argument(
        "--epochs", type=int, default=100, help="num of training epochs"
    )
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument(
        "--save_file", action="store_true", default=False, help="save the script"
    )
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="the explore rate in the gradient descent process",
    )
    parser.add_argument(
        "--train_portion", type=float, default=0.5, help="portion of training data"
    )
    parser.add_argument(
        "--unrolled",
        action="store_true",
        default=False,
        help="use one-step unrolled validation loss",
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="temperature of AGLayer"
    )
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=0.08,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate",
        type=float,
        default=0.005,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for arch encoding",
    )
    parser.add_argument(
        "--gnn0_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument(
        "--pooling_ratio", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument("--beta", type=float, default=5e-3, help="global pooling ratio")
    parser.add_argument("--gamma", type=float, default=5.0, help="global pooling ratio")
    parser.add_argument("--eta", type=float, default=0.1, help="global pooling ratio")
    parser.add_argument(
        "--eta_max", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument(
        "--with_conv_linear",
        type=bool,
        default=False,
        help=" in NAMixOp with linear op",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="num of layers of GNN method."
    )
    parser.add_argument(
        "--withoutjk", action="store_true", default=False, help="remove la aggregtor"
    )
    parser.add_argument(
        "--search_act",
        action="store_true",
        default=False,
        help="search act in supernet.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--BN", type=int, default=64, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--graph_dim", type=int, default=8, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="default hidden_size in supernet",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--num_sampled_archs", type=int, default=5, help="sample archs from supernet"
    )

    # for ablation stuty
    parser.add_argument(
        "--remove_pooling",
        action="store_true",
        default=False,
        help="remove pooling block.",
    )
    parser.add_argument(
        "--remove_readout",
        action="store_true",
        default=False,
        help="exp5, only search the last readout block.",
    )
    parser.add_argument(
        "--remove_jk",
        action="store_true",
        default=False,
        help="remove ensemble block, Graph representation = Z3",
    )

    # in the stage of update theta.
    parser.add_argument(
        "--temp", type=float, default=0.2, help=" temperature in gumble softmax."
    )
    parser.add_argument(
        "--loc_mean",
        type=float,
        default=10.0,
        help="initial mean value to generate the location",
    )
    parser.add_argument(
        "--loc_std",
        type=float,
        default=0.01,
        help="initial std to generate the location",
    )
    parser.add_argument(
        "--lamda",
        type=int,
        default=2,
        help="sample lamda architectures in calculate natural policy gradient.",
    )
    parser.add_argument(
        "--adapt_delta",
        action="store_true",
        default=False,
        help="adaptive delta in update theta.",
    )
    parser.add_argument(
        "--delta", type=float, default=1.0, help="a fixed delta in update theta."
    )
    parser.add_argument(
        "--w_update_epoch", type=int, default=1, help="epoches in update W"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="darts",
        help="how to update alpha",
        choices=["mads", "darts", "snas"],
    )

    args = parser.parse_args()
    torch.set_printoptions(precision=4)

    return args


if __name__ == "__main__":
    set_seed(0)

    hps = {
        "num_layers": 2,
        "learning_rate": 0.00034828472005404485,
        "learning_rate_min": 0.00019242101475226765,
        "weight_decay": 0,
        "temperature": 4.089492969843236,
        "arch_learning_rate": 0.0003948218327378405,
        "arch_weight_decay": 0.001,
        "gnn0_learning_rate": 0.03391343886431106,
        "gnn0_weight_decay": 0,
        "pooling_ratio": 0.3029329352563719,
        "dropout": 0.2623320360058418,
        "beta": 0.00462003423626971,
        "eta": 0.06839360891312493,
        "eta_max": 0.5871192734433861,
        "gamma": 0.010494136340498105,
    }

    args = parser_args()

    for k, v in hps.items():
        setattr(args, k, v)

    data, num_nodes = load_data(args.data, batch_size=args.batch_size)

    num_features = data[0].num_features
    num_classes = data[0].num_tasks
    # OneShotOGBEstimator
    # 进行训练，仅通过给定的模型和数据来进行推理并计算评估指标
    estimator = OneShotOGBEstimator(
        loss_f="binary_cross_entropy_with_logits", evaluation=[Auc()]
    )
    space = GracesSpace(
        input_dim=num_features,
        output_dim=num_classes,
        num_nodes=num_nodes,
        mol=True,  # for ogbg
        virtual=True,  # for ogbg
        criterion=torch.nn.BCEWithLogitsLoss(),  # for ogbg
        args=args,
    )

    space.instantiate()
    algo = Graces(num_epochs=args.epochs, args=args)
    algo.search(space, data, estimator)
