import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "annotation",
        type=Path,
        help="path to annotation csv file or parent directory",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="path to label txt file including label names separated by newline",
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        type=Path,
        help="path to directory converted annotation file",
    )
    parser.add_argument(
        "--suffix",
        default="csv",
        type=str,
        help="annotation file suffix. csv or txt",
    )
    parser.add_argument(
        "--sep",
        default=",",
        type=str,
        help="delimiter/separator of annotation file. comma or space",
    )
    return parser.parse_args()


def main():
    args = parser_args()

    if str(args.annotation).endswith(args.suffix):
        annotations = [args.annotation]
    else:
        annotations = sorted(
            [
                anno
                for anno in args.annotation.rglob(f"*.{args.suffix}")
                if "mmaction" not in anno.name
            ]
        )

    if args.labels is not None:
        label2index = read_label2index_map(args.labels)

    with tqdm(annotations) as pbar:
        for anno in pbar:
            pbar.set_description(f"processing {anno.name}")
            df = pd.read_csv(anno, sep=args.sep)
            # ignore not annotation csv
            if len(set(df.columns) & set(["start-frame", "phase-id"])) == 0:
                pbar.set_postfix_str("skip")
                continue
            df = convert_start_phase2start_end_phase(df)

            if args.labels is not None:
                df["phase-id"] = df["phase-id"].map(label2index)

            if args.save_dir is None:
                save_dir = anno.parent
            else:
                save_dir = args.save_dir
                save_dir.mkdir(parents=True, exist_ok=True)
            # df.to_csv(save_dir / f"{anno.stem}.txt", index=False, header=False, sep=" ")
            df.to_csv(save_dir / "mmaction.txt", index=False, header=False, sep=" ")


def read_label2index_map(label_txt):
    with label_txt.open() as fr:
        return {label.strip(): i for i, label in enumerate(fr)}


def convert_start_phase2start_end_phase(df_anno):
    df_anno["end-frame"] = df_anno["start-frame"].shift(-1) - 1
    df_anno = df_anno[~df_anno["phase-id"].isin(["end", "e"])]
    df_anno["end-frame"] = df_anno["end-frame"].astype(int)
    return df_anno[["start-frame", "end-frame", "phase-id"]]


if __name__ == "__main__":
    main()
