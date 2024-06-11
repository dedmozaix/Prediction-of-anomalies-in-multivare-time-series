from test_tube import HyperOptArgumentParser


class HparamsDefiner:
    def __init__(self, data_name, num_epochs=5, pretrained_model=False, pretrained_model_path=None,
                 pred_horizon=200, train_horizon=1200, local=3,
                 n_kernels=32, w_kernel=1, dropout=0.1,
                 window=64, criterion="l1",
                 lr=0.0005, batch_size=16, idj_maximum=20):

        self.parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

        # required args
        self.parser.add_argument('--data_name', default=data_name, type=str)

        self.parser.add_argument('--num_epochs', default=num_epochs, type=int)
        self.parser.add_argument('--pretrained_model', default=pretrained_model, type=bool)
        self.parser.add_argument('--pretrained_model_path', default=pretrained_model_path, type=str)
        self.parser.add_argument('--pred_horizon', default=pred_horizon, type=int)
        self.parser.add_argument('--train_horizon', default=train_horizon, type=int)

        # network params
        self.parser.opt_list('--local', default=local, options=[3, 5, 7], type=int, tunable=True)
        self.parser.opt_list('--n_kernels', default=n_kernels, options=[32, 50, 100], type=int, tunable=True)
        self.parser.add_argument('-w_kernel', type=int, default=w_kernel)
        self.parser.opt_list('--d_model', type=int, default=512, options=[512], tunable=False)
        self.parser.opt_list('--d_inner', type=int, default=2048, options=[2048], tunable=False)
        self.parser.opt_list('--d_k', type=int, default=64, options=[64], tunable=False)
        self.parser.opt_list('--d_v', type=int, default=64, options=[64], tunable=False)
        self.parser.opt_list('--n_head', type=int, default=8, options=[8], tunable=False)
        self.parser.opt_list('--n_layers', type=int, default=6, options=[6], tunable=False)
        self.parser.opt_list('--drop_prob', type=float, default=dropout, options=[0.1, 0.2, 0.5], tunable=False)

        self.parser.opt_list('--window', default=window, type=int, options=[32, 64, 128], tunable=True)

        # training params (opt)
        self.parser.opt_list('--learning_rate', default=lr, type=float,
                             options=[0.0001, 0.0005, 0.001, 0.005, 0.008],
                             tunable=True)
        self.parser.opt_list('--optimizer_name', default='adam', type=str, options=['adam'], tunable=False)
        self.parser.opt_list('--criterion', default=criterion, type=str, options=['l1', 'mse'], tunable=False)

        self.parser.opt_list('--batch_size', default=batch_size, type=int, options=[16, 32, 64, 128, 256],
                             tunable=False)

        # only MARIMA params
        self.parser.opt_list('--idj_maximum', default=idj_maximum, type=int, options=[5, 10, 15, 20], tunable=True)

    def get_parser(self):
        return self.parser
