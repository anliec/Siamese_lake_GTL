import argparse
import pickle
from keras.models import load_model

from evaluator.evaluate import DatasetTester


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path',
                        required=True,
                        type=str,
                        dest="model_path")
    parser.add_argument('-o', '--output-file',
                        default="out.pickle",
                        type=str,
                        dest="out_path")
    parser.add_argument('-d', '--dataset-path',
                        required=True,
                        type=str,
                        dest="dataset_path")
    args = parser.parse_args()

    print("Loading model")
    model = load_model(args.model_path)

    print("Init evaluator")
    evaluator = DatasetTester(args.dataset_path)

    print("Evaluating the model on the localisation dataset")
    results_list = evaluator.evaluate(model, mode='localisation', add_coordinate=False)

    print("Write the score list to pickle file:", args.out_path)
    with open(args.out_path, 'wb') as handle:
        pickle.dump(results_list, handle)


if __name__ == '__main__':
    main()
