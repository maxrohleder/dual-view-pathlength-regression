import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dual-view pathlength regression')
    parser.add_argument('--data', action='store', type=Path, required=True)
    parser.add_argument('--testdata', action='store', type=Path, required=True)
    parser.add_argument('--results', action='store', type=Path, required=True)

    parser.add_argument('--epochs', action='store', default=100, type=int, required=False)
    parser.add_argument('--bs', action='store', default=1, type=int, required=False)
    parser.add_argument('--lr', action='store', default=1e-3, type=float, required=False)
    parser.add_argument('--workers', action='store', default=4, type=int, required=False)