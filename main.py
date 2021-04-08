import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

from utils import path_constants

DATASET_PICKLE_PATHS = {
    "FGNET": path_constants.FGNET_PICKLE,
    "WIKI": path_constants.WIKI_PICKLE,
    "UTK": path_constants.UTK_PICKLE,
}

def search_model_file(model_fname):
    for root, __dirs, files in os.walk(path_constants.MODEL_HISTORY_SAVE):
        for fname in files:
            if fname==model_fname:
                return os.path.join(root, fname)

parser = argparse.ArgumentParser(
    description="NN Project: re-implementation of C3AE", 
    epilog="Reference PaperWithCode: https://paperswithcode.com/paper/c3ae-exploring-the-limits-of-compact-model"
)

parser.add_argument(
    "-dataprocessing",
    help="Process a dataset into pickle form, so it can be used by the other modules. "
         "You need to provide a dataset name: possile choices right now are fgnet and wiki, "
         "but check DATASET_PATHS in the source code to be sure"
)

parser.add_argument(
    "-training",
    help="Train a new model on a processed dataset. "
         "You need to provide a dataset name: possile choices right now are fgnet and wiki, "
         "but check DATASET_PATHS in the source code to be sure"
)

parser.add_argument(
    "-single-prediction",
    help="Predict the age of one person by providing the path of a photo. "
         "Also specify the -m argument."
)

parser.add_argument(
    "-use-test-set",
    help="Use the given dataset as a test set for a trained model. "
         "Also specify the -m argument."
)



parser.add_argument(
    '-e', '--epochs',
    default=100,
    type=int,
    help='Number of epochs to train'
)

parser.add_argument(
    '-m', '--model',
    help="Name of the model to use in certain modes. Provide only the filename, and we'll look for it in the models folder"
)

parser.add_argument(
    '--silent',
    action="store_true",
    help="In -single-prediction, only output the age and do nothing more."
)

parser.add_argument(
    '--ablation',
    action="store_true",
    help="In -training, start an ablation study that trains many variants of the model."
)

parser.add_argument(
    '--ablate-context', '--abco',
    action="store_true",
    help="In -use-test-set, activate context ablation."
)

parser.add_argument(
    '--ablate-cascade', '--abca',
    action="store_true",
    help="In -use-test-set, activate cascade ablation."
)

args = parser.parse_args()

# importiamo i moduli all'ultimo momento per evitare il caricamento di tensorflow se sbagliamo la CLI

# accendiamo una flag per gestire meglio la GPU e permetter a tutto di funzionare
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    

if args.dataprocessing!=None:
    name = args.dataprocessing.lower()
    from master import dataprocessing
    import tensorflow as tf
    with tf.device('/gpu:0'):   # or '/cpu:0'
        dataprocessing.process_dataset(name)

elif args.training!=None:
    dataset_name = args.training.upper()
    epochs = args.epochs
    dataset_pickle_path = DATASET_PICKLE_PATHS[dataset_name]
    if args.model:
        model_path = search_model_file(args.model)
    else:
        model_path=None
    from master import training
    training.train_main(dataset_pickle_path, epochs, args.ablation, model_path)

elif args.single_prediction!=None:
    image_path = args.single_prediction
    model_path = search_model_file(args.model)
    from tests import single_prediction
    single_prediction.sp_main(image_path, model_path, args.silent)

elif args.use_test_set!=None:
    dataset_name = args.use_test_set.upper()
    dataset_pickle_path = DATASET_PICKLE_PATHS[dataset_name]
    model_path = search_model_file(args.model)
    from tests import use_test_set
    use_test_set.test_main(dataset_pickle_path, model_path, args.ablate_context, args.ablate_cascade)
