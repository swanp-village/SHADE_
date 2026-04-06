import argparse

from MRR.evaluator.train import show_data, train_evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-data", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = vars(parser.parse_args())
    skip_plot = args["skip_plot"]
    if args["show_data"]:
        show_data(skip_plot)
    else:
        train_evaluator()
