import argparse

import path_constants

DATASET_PICKLE_PATHS = {
    "FGNET": path_constants.FGNET_PICKLE,
    "WIKI": path_constants.WIKI_PICKLE,
}

parser = argparse.ArgumentParser(
    description="NN Project: re-implementation of C3AE", 
    epilog="Reference PaperWithCode: https://paperswithcode.com/paper/c3ae-exploring-the-limits-of-compact-model"
)

parser.add_argument(
    "dataset",
    help="Select dataset. "
         "Possible choices right now are fgnet and wiki, "
         "but check DATASET_PATHS in the source code to be sure"
)

parser.add_argument(
    '-e', '--epochs',
    default=100,
    type=int,
    help='Number of epochs to train'
)

args = parser.parse_args()

dataset_name = args.dataset.upper()
epochs = args.epochs

dataset_pickle_path = DATASET_PICKLE_PATHS[dataset_name]

# importiamo training all'ultimo momento per evitare il caricamento di tensorflow se sbagliamo la CLI
import training
training.train_main(dataset_pickle_path, epochs)
