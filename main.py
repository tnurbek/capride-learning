import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_cifar100_test_loader, get_cifar100_train_loader, get_cifar10_train_loader, get_cifar10_test_loader, \
    get_mnist_train_loader, get_mnist_test_loader, get_ham10000_train_loader, get_ham10000_test_loader 


def main(config):
    
    prepare_dirs(config)

    # ensure reproducibility
    kwargs = {}
    if config.use_gpu:
        # torch.cuda.manual_seed_all(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory}
        # torch.backends.cudnn.deterministic = True

    # instantiate data loaders
    test_data_loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, **kwargs
    )

    if config.is_train:
        train_data_loader = get_cifar10_train_loader(
            config.data_dir, config.batch_size,
            config.random_seed, config.shuffle, **kwargs
        )
        data_loader = (train_data_loader, test_data_loader)
    else:
        data_loader = test_data_loader

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
