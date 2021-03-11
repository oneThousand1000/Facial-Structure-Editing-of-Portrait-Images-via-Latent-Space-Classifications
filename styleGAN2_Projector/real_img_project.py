from stylegan2 import my_project
import argparse

import re
_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''
def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

    Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl',
                                                 required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds',
                                                 default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int,
                                                 help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float,
                                                 help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir',
                                                 help='Root directory for run results (default: %(default)s)',
                                                 default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl',
                                            required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int,
                                            help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int,
                                            help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir',
                                            help='Root directory for run results (default: %(default)s)',
                                            default='results', metavar='DIR')
    project_real_images_parser.add_argument('--save-dir', help='Root directory for run results (default: %(default)s)',
                                            dest='save_dir', required=True)

    args = parser.parse_args()
    my_project.run(args)


if __name__ == "__main__":
    main()
