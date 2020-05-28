import argparse
from mask import mask


parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--resume', default='cp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='data/Human6', help='videos or image files')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
parser.add_argument('--mask_coord', default=None, type=str, required=False, help='mask-coordination x, y, w, h')
args = parser.parse_args()

mask(args)
