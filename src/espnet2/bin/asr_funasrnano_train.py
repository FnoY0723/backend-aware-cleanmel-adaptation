
from espnet2.tasks.asr_funasrnano import ASRTask


def get_parser():
    return ASRTask.get_parser()


def main(cmd=None):
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
