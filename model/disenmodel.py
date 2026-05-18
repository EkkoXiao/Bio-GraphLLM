import torch
from model.nas.gencoder import RGEncoder, CrossAttention
from model.nas.archgen import AG, InvDisenHead
from model.nas.supernet import Network

from autogllight.nas.space import BaseSpace
from autogllight.nas.space.graces_space.genotypes import NA_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES

class DisenModel(BaseSpace):
    def __init__(self, input_dim, env_dim, mol, virtual, args, use_forward):
        super().__init__()
        self.input_dim = input_dim
        self.env_dim = env_dim
        self.mol = mol
        self.virtual = virtual
        self.args = args
        self.use_forward = use_forward
        self.build_graph()

    def build_graph(self):
        self.encoder = RGEncoder(
            in_dim=self.input_dim,
            hidden_size=self.args.graph_dim,
            num_layers=2,
            dropout=0.5,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            mol=self.mol,
            virtual=self.virtual,
        )
        self.cross_attn = CrossAttention(
            drug_dim=self.args.graph_dim,
            target_dim=self.args.target_dim,
            hidden_dim=self.args.cross_attn_dim,
        )
        self.supernet = Network(
            in_dim=self.env_dim,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            mol=False,  # with environmental data, not molecule
            virtual=self.virtual,
        )
        num_na_ops = len(NA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)
        self.ag = AG(args=self.args, num_op=num_na_ops, num_pool=num_pool_ops)
        self.invdisenhead = InvDisenHead(input_dim=self.args.graph_dim, invariant_dim=self.args.invariant_dim, variant_dim=self.args.variant_dim)
        self.explore_num = 0

    def forward(self, data, envdata, targets):
        if not self.use_forward:
            return self.prediction
        # graph_emb0 [batch_size * 8]
        # sslout [batch_size * 3]
        graph_emb0 = self.encoder(data, mode="mixed")
        # graph_emb1 [batch_size * 8]
        graph_emb1 = self.cross_attn(graph_emb0, targets)
        # graph_emb1 [batch_size * 8]
        graph_emb, disen_loss = self.invdisenhead(graph_emb0, graph_emb1)
        # graph_alpha [num_layers * batch_size * 6]，代表每层 6 个 options
        graph_alpha, cosloss = self.ag(graph_emb)
        # pred [batch_size * 1]
        # emb [batch_size * 128]
        if self.args.env:
            node_embs, node_mask = self.supernet(envdata, mode="mads", graph_alpha=graph_alpha)
        else:
            data.x = data.x.float()
            node_embs, node_mask = self.supernet(data, mode="mads", graph_alpha=graph_alpha)
            data.x = data.x.long()
        return disen_loss, cosloss, node_embs, node_mask

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graces")

        parser.add_argument(
            "--supernet_learning_rate", type=float, default=0.01, help="init learning rate"
        )
        parser.add_argument(
            "--supernet_learning_rate_min", type=float, default=0.001, help="min learning rate"
        )
        parser.add_argument("--supernet_weight_decay", type=float, default=5e-4, help="weight decay")
        parser.add_argument(
            "--epsilon",
            type=float,
            default=0.0,
            help="the explore rate in the gradient descent process",
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
            "--encoder_learning_rate",
            type=float,
            default=0.005,
            help="learning rate for arch encoding",
        )
        parser.add_argument(
            "--encoder_learning_rate_min",
            type=float,
            default=0.0,
            help="minimum learning rate for arch encoding",
        )
        parser.add_argument(
            "--encoder_weight_decay",
            type=float,
            default=1e-3,
            help="weight decay for arch encoding",
        )
        parser.add_argument(
            "--pooling_ratio", type=float, default=0.5, help="global pooling ratio"
        )
        parser.add_argument("--beta", type=float, default=5e-3, help="cosloss weight")
        parser.add_argument("--gamma", type=float, default=5e-3, help="sslloss weight")
        parser.add_argument("--eta", type=float, default=0.1, help="errorloss weight")
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
            "--search_act",
            action="store_true",
            default=False,
            help="search act in supernet.",
        )
        parser.add_argument(
            "--hidden_size", type=int, default=128, help="default hidden_size in supernet"
        )
        parser.add_argument(
            "--graph_dim", type=int, default=8, help="default dim in encoder"
        )
        parser.add_argument(
            "--target_dim", type=int, default=1280, help="default dim for targets"
        )
        parser.add_argument(
            "--cross_attn_dim", type=int, default=64, help="default dim for cross attention"
        )
        parser.add_argument(
            "--invariant_dim", type=int, default=2, help="default dim for invariant representation"
        )
        parser.add_argument(
            "--variant_dim", type=int, default=6, help="default dim for variant representation"
        )
        parser.add_argument(
            "--dropout", type=float, default=0.5, help="default dropout in supernet"
        )
        # in the stage of update theta.
        parser.add_argument(
            "--temp", type=float, default=0.2, help="temperature in gumble softmax."
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
            "--model_type",
            type=str,
            default="darts",
            help="how to update alpha",
            choices=["mads", "darts", "snas"],
        )
        parser.add_argument(
            "--temperature", type=float, default=1, help="temperature of AGLayer"
        )
        return parent_parser