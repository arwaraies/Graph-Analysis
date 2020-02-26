import argparse
import os
from itertools import chain

import attr

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train

#list of file names used for training and testing
FILENAMES = {
    'train': 'target-kg/v01/train.txt',
    'valid': 'target-kg/v01/valid.txt',
    'test': 'target-kg/v01/test.txt',
}


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Example on FB15k')
    parser.add_argument('--config', default='examples/configs/complex_config.py',
                        help='Path to config file')
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--data_dir', default='data',
                        help='where to save processed data')
    parser.add_argument('--no-filtered', dest='filtered', action='store_false',
                        help='Run unfiltered eval')

    params = ['checkpoint_path=model/model','num_epochs=10','lr=0.1','dimension=128','comparator=dot','loss_fn=softmax','margin=0.1']
    
    param_values = []
    param_values.append(params)

    count = 0
            
    parser.set_defaults(config='examples/configs/translation_config.py')
                                                                     
    args = parser.parse_args()
    args.param = [params]

    if args.param is not None:
        overrides = chain.from_iterable(args.param)
    else:
        overrides = None
    
    data_dir = args.data_dir

    edge_paths = [os.path.join(data_dir, name) for name in FILENAMES.values()]
    convert_input_data(
        args.config,
        edge_paths,
        lhs_col=0,
        rhs_col=1,
        rel_col=2,
    )
    
    config = parse_config(args.config, overrides)
    train_path = [convert_path(os.path.join(data_dir, FILENAMES['train']))]
    train_config = attr.evolve(config, edge_paths=train_path)
    
    train(train_config)
    
    eval_path = [convert_path(os.path.join(data_dir, FILENAMES['valid']))]
    eval_config = attr.evolve(config, edge_paths=eval_path)
    do_eval(eval_config)
    
if __name__ == "__main__":
    main()
