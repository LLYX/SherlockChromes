import logging
import os
import pprint
import sys
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from importlib.util import find_spec, spec_from_file_location, module_from_spec

# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                     level=logging.DEBUG,
#                     stream=sys.stdout)


def create_obj_from_cfg_section(cfg, section_name):
    if section_name in cfg:
        path = cfg[section_name]['script_path']
        mod_name = cfg[section_name]['module_name']
        spec = spec_from_file_location(mod_name, path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[mod_name] = mod

        if 'kwargs' in cfg[section_name]:
            obj_kwargs = cfg[section_name]['kwargs']
            obj = getattr(mod, cfg[section_name]['name'])(**obj_kwargs)
        else:
            obj = getattr(mod, cfg[section_name]['name'])()

        return obj
    return None


def run_experiment(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration to verify params
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)

    assert 'dataset' in cfg, 'dataset is required!'
    assert 'model' in cfg, 'model is required!'
    assert 'train' in cfg, 'train is required!'

    if 'general' in cfg:
        device = cfg['general']['device']
    else:
        device = 'cpu'

    transform = create_obj_from_cfg_section(cfg, 'transform')

    dataset = create_obj_from_cfg_section(cfg, 'dataset')
    dataset.transform = transform

    model = create_obj_from_cfg_section(cfg, 'model')

    if 'loss' in cfg:
        loss = create_obj_from_cfg_section(cfg, 'loss')
    else:
        loss = None

    optimizer_kwargs = cfg['optimizer']['kwargs']

    lr_scheduler_kwargs = cfg['scheduler']['kwargs']

    sampling_fn = create_obj_from_cfg_section(cfg, 'sampling_fn')

    collate_fn = create_obj_from_cfg_section(cfg, 'collate_fn')

    train_path = cfg['train']['script_path']
    sys.path.insert(1, os.path.dirname(train_path))
    train_spec = spec_from_file_location('train', train_path)
    train_mod = module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)
    train_kwargs = cfg['train']['kwargs']

    train_mod.main(
        dataset,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        train_kwargs,
        device
    )


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): The absolute path to the directory of script.
        cfg (dict): Dictionary of config params.

    Output:
        cfg (dict): Dictionary of config params with absolute paths.
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])

    return cfg


def extract_module_names(cfg):
    """
    Extract all module names based on filename from script paths.

    Args:
        cfg (dict): Dictionary of config params.

    Output:
        cfg (dict): Dictionary of config params with added module names.
    """
    paths = []
    for key1 in cfg.keys():
        for key2 in cfg[key1]:
            if key2.endswith("_path"):
                paths.append((key1, key2))

    for path in paths:
        cfg[path[0]]['module_name'] = os.path.splitext(
            os.path.basename(cfg[path[0]][path[1]]))[0]

    return cfg


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Args:
        yaml_filepath (str): Path to yaml config file.

    Output:
        cfg (dict): Dictionary of config params.
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.full_load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    cfg = extract_module_names(cfg)

    return cfg


def get_parser():
    """Get parser object."""
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)

    parser.add_argument("-p", "--path",
                        dest="path",
                        help="relative path to project root",
                        metavar="PATH",
                        required=True)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    sys.path.insert(0, args.path)
    run_experiment(args.filename)
