import argparse
import os
import tempfile
import datetime as dt
import learner
import traceback

def main():
	"""
    Main method to be called by __main__. Contains CLI handling and passes the variables to learner.py
    """
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
	parser.add_argument('--lr_ramp_on', help='turn lr ramp on on', action = "store_true", default=False)
	parser.add_argument('--synth_first', help='synthetic data first to learn', action = "store_true", default=False)
	parser.add_argument('--nat_first', help='synthetic data first to learn', action = "store_true", default=False)
	parser.add_argument('--startup_fraction', help='startup fraction of the training epochs', type=float, default=1e-1)
	parser.add_argument('--inter_dim_i', help='size of intermediate layer (before and after bottleneck)', type=int, default=512)
	parser.add_argument('--bneck_i', help='size of bottleneck', type=int, default=256)
	parser.add_argument('--d_l_f', help='decoder loss weight factor', type=float, default=1e3)
	parser.add_argument('--r_l_f', help='regressor loss weight factor', type=float, default=1)
	parser.add_argument('--KLD_l_f', help='Kullback-Leibler divergence factor', type=float, default=1)
	parser.add_argument('--directory', help='place output in this directory', type=str, default=dt.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
	parser.add_argument('--checkpoint', help='resume from this checkpoint', type=str, default=None)  # e.g. 10000.pth
	parser.add_argument('--epochs', help='stop after this many epochs', type=int, default=10)
	parser.add_argument('--regressor_start', help='regressor loss starts with this epoch', type=int, default=0)
	parser.add_argument('--plot_interval', help='plots intermediate results every x epochs', type=int, default=1000)
	parser.add_argument('--test_interval', help='test architecture every x epochs', type=int, default=10)
	parser.add_argument('--eval_interval', help='sample images from repsentational space every x epochs', type=int, default=1000)
	parser.add_argument('--checkpoint_interval', help='genereates a checkpoint every x epochs', type=int, default=5)
	parser.add_argument('--delete_checkpoints', help='delete old checkpoints', action="store_false", default=True)
	parser.add_argument('--transfer', help='try transfer learning', action="store_true", default=False)
	parser.add_argument('--freeze_partly', help='partial network freezing during transfer learning', action="store_true", default=False)
	parser.add_argument('--freeze_outer', help='whether to freeze the inner or outer layers(default false)', action="store_true", default=False)
	parser.add_argument('--thaw', help='thawing the frozen layers during transfer learning', action="store_true", default=False)
	parser.add_argument('--weight_decay', help='weight decay rate', type=float, default=1e-4)
	parser.add_argument('--dropout_enc', help='dropout for the encoder', type=float, default=0.1)
	parser.add_argument('--dropout_fc', help='dropout for the fully connected layers', type=float, default=0.1)
	parser.add_argument('--leak_enc', help='leakiness of LReLUs in encoder', type=float, default=0.2)
	parser.add_argument('--leak_dec', help='leakiness of LReLUs in decoder', type=float, default=0.2)
	parser.add_argument('--leak_regr', help='leakiness of LReLUs in regressor', type=float, default=0.1)
	parser.add_argument('--bs', help='batch size', type=int, default=128)
	parser.add_argument('--s', help='seed for reproduction', type=int, default=1)
	parser.add_argument('--data_case', help='either bright-field (bf) or phase-contrast (pc)', type=str, default="pc")
	parser.add_argument('--n', help='amount of noise to vary images', type=float, default=0.0025)
	parser.add_argument('--crop_size', help='crop size to vary images', type=float, default=0.9)
	parser.add_argument('--cross_feed', help='whether or not to feed decoded syn data into the nat train data', action="store_true", default=False)
	parser.add_argument('--dyn_data', help='generate new syn data or not', action="store_true", default=False)
	parser.add_argument('--ratio', help='multiplier of generated data', type=float, default=1.0)
	#parser.add_argument('--nat_only', help='whether to use syn data or not', action="store_true", default=False)

	args = parser.parse_args()

	try:
		learner.main(args.lr, args.inter_dim_i, args.bneck_i, args.d_l_f, args.r_l_f, args.KLD_l_f, args.directory, args.checkpoint, args.epochs,
					 args.regressor_start, args.plot_interval, args.test_interval, args.eval_interval, args.checkpoint_interval, args.delete_checkpoints, args.weight_decay,
					 args.dropout_enc, args.dropout_fc, args.leak_enc, args.leak_dec, args.leak_regr, args.bs, args.s, args.n, args.transfer, args.thaw, args.crop_size, args.data_case, args.lr_ramp_on, args.synth_first,args.nat_first, args.startup_fraction, args.cross_feed, args.dyn_data, args.ratio, args.freeze_partly, args.freeze_outer)

	except RuntimeError as e:
		learner.playSound()
		traceback.print_exc()

if __name__ == '__main__':
	main()
