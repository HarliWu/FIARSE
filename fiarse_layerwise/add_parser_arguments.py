from argparse import ArgumentParser

def new_arguements(parser: ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr-mask', type=float, default=0.1, help='learning rate for mask')
    parser.add_argument('--lr-global', type=float, default=1.0, help='global learning rate')
    parser.add_argument('--lr-diminish', action='store_true', help='diminish the learning rate')
    parser.add_argument('--lr-decay', type=int, default=10, help='number of communication rounds diminish the learning rate')
    parser.add_argument('--T', type=int, default=100, help='Communication rounds')

    # Setting for local iterations 
    parser.add_argument('--K', type=int, default=5, help='Local iterations/epochs')
    parser.add_argument('--K-unit', type=str, default='iterations', help='iterations, epochs, total_size')

    # Setting for the pruning model 
    parser.add_argument('--model-size', action='extend', nargs='+', help='The size of the model')
    parser.add_argument('--model-dist', action='extend', nargs='+', help='How the model size is different')
    
    # Save checkpoints 
    parser.add_argument('--save-freq', type=int, default=20, help='save check point every k rounds')