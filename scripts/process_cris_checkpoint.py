import torch
from torch.serialization import FILE_LIKE


def main(
    checkpoint_input_path: FILE_LIKE,
    checkpoint_output_path: FILE_LIKE,
    prefix: str,
    pickle_protocol: int,
):
    checkpoint = torch.load(checkpoint_input_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]

    is_valid_checkpoint = all(k for k in state_dict if k.startswith(prefix))

    if not is_valid_checkpoint:
        raise ValueError(
            f"Invalid checkpoint. All the keys of state_dict must start with `{prefix}`"
        )

    key_start_idx = len(prefix) + 1
    new_state_dict = {k[key_start_idx:]: v for k, v in state_dict.items()}

    torch.save(new_state_dict, checkpoint_output_path, pickle_protocol=pickle_protocol)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="A script to convert the distributed checkpoint to a single machine checkpoint."
    )

    parser.add_argument(
        "--checkpoint-input-path",
        type=str,
        default="pretrain/cris_best.pth",
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--checkpoint-output-path",
        type=str,
        default="pretrain/cris_best_single.pth",
        help="Path to save the converted checkpoint.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="model.",
        help="The prefix of the state_dict in the checkpoint.",
    )
    parser.add_argument(
        "--pickle-protocol",
        type=int,
        default=5,
        help="The protocol to use when pickling the checkpoint.",
    )

    args = parser.parse_args()

    main(**vars(args))
