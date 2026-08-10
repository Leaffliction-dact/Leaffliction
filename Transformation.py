import argparse

help_msg_show = """display in a window the transformations preformed on the
                images."""
help_msg_transform = """transform a image folder with the diferent extraction
                        methods."""


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    parser_show = subparsers.add_parser("show", help=help_msg_show)
    parser_show.add_argument("img", type=str, help="image to be transformed.")

    parser_transform = subparsers.add_parser("transform",
                                             help=help_msg_transform)
    parser_transform.add_argument("--src",
                                  metavar=('src_directory'),
                                  type=str,
                                  required=True,
                                  help="image folder to be transformed.")
    parser_transform.add_argument("--dst",
                                  metavar=('dst_directory'),
                                  type=str,
                                  required=True,
                                  help="""destination directory for the
                                        transformed images.""")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
