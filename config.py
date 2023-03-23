import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='CaPriDe Learning')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name): 
    arg = parser.add_argument_group(name)
    arg_lists.append(arg) 
    return arg 
 
 
# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--num_classes', type=int, default=10,
                      help='Number of classes to classify')
data_arg.add_argument('--batch_size', type=int, default=128,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--pin_memory', type=str2bool, default=True,
                      help='whether to copy tensors into CUDA pinned memory')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train indices')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum value')
train_arg.add_argument('--epochs', type=int, default=1,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.1,
                       help='Initial learning rate value')
train_arg.add_argument('--weight_decay', type=float, default=5e-4,
                       help='value of weight dacay for regularization')
train_arg.add_argument('--nesterov', type=str2bool, default=True,
                       help='Whether to use Nesterov momentum')
train_arg.add_argument('--gamma', type=float, default=0.1,
                       help='value of learning rate decay')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=False,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/cifar10',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',  
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/', 
                      help='Directory in which Tensorboard logs will be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--save_name', type=str, default='capride_c10_r18_p10_imb2', 
                      help='Name of the model to save') 
misc_arg.add_argument('--model_num', type=int, default=5, 
                      help='Number of CaPriDe models to train')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
