import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    seed_override = args.seed
    args = vars(args)
    args.update(param)
    for item in args.pop("set") or []:
        key, _, value = item.partition("=")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # plain string, e.g. prefix=entropy
        args[key] = value
    if seed_override is not None:
        args["seed"] = [seed_override]

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./exps/prism_ina.json',
                        help='Json file of settings.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed (overrides the config seed list).')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a config key, e.g. --set router_mode=task '
                             '--set prefix=rtask (value parsed as JSON, else string).')
    return parser

if __name__ == '__main__':
    main()
