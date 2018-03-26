import argparse

parser = argparse.ArgumentParser()

# Required path files
parser.add_argument('--gaze_path', default='/home/akhil/Downloads/Warped_Gaze/', help=' Path to the Gaze Files ')
parser.add_argument('--learning_rate', default=0.001, help='Learning Rate')
parser.add_argument('--epochs', default=10, help='Number of Epochs')
parser.add_argument('--iterations', default=100000, help='Number of iterations')
parser.add_argument('--dimension2', default=7500, help='Number of columns in D matrix')
parser.add_argument('--sparsity_coeff', default=0.14, help='Coefficient of Sparsity')
parser.add_argument('--save_dir', default='/home/akhil/PycharmProjects/Sparse_Coding_with_clustered_data/saved_models/', help=' Path to saved models')
parser.add_argument('--n_trainable', default=5, help='Number of trainable elements in a particular column')

args = parser.parse_args()