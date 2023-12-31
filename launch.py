
from rlpyt.utils.launching.arguments import get_args
from rlpyt.utils.launching.launcher import start_experiment
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


if __name__ == "__main__":

    args = get_args()
    start_experiment(args) # launches the actual experiment inside of a tmux session









