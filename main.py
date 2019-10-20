import argparse

from training.train import main as run_train
from testing.test import main as run_test
from testing.inference import main as run_inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument("--config", type=str)

    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=run_test)
    test_parser.add_argument("--config", type=str)

    infer_parser = subparsers.add_parser("inference")
    infer_parser.set_defaults(func=run_inference)
    infer_parser.add_argument("--config", type=str)
    infer_parser.add_argument("--img", type=str, required=True)
    infer_parser.add_argument("--viz", action='store_true', help='whether to display input image')

    args = parser.parse_args()
    args.func(args)
