# Train model on Platform 2D environment
import sys
sys.path.append('./') # allows it to be run from parent dir
import utils.experiment as utils


if __name__ == '__main__':
    parser = utils.experiment_argparser()
    args = parser.parse_args()
    train_config = utils.train_params_from_args(args)
    utils.run_training(train_config)