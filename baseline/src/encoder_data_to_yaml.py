from pathlib import Path
import pickle

import click


@click.command()
@click.argument('encoder_data_pkl')
def main(encoder_data_pkl: str):
    # encoder_data_pkl = 'encoder_data.xlc.s262-t94.pkl'
    with open(encoder_data_pkl, 'rb') as f:
        obj = pickle.load(f)

    encoder_data_yaml = Path(encoder_data_pkl).with_suffix('.yaml')
    with open(encoder_data_yaml, 'w') as f:
        yaml.dump(
            dict(
                segmented = obj.segmented,
                PAD_ID = obj.PAD_ID,
                SEP_ID = obj.SEP_ID,
                BOS_ID = obj.BOS_ID,
                EOS_ID = obj.EOS_ID,
                vocabulary=obj.vocabulary,
            ), f)

if __name__ == "__main__":
    main()
