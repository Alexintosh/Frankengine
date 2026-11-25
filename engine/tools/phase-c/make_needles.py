"""Create live retrieval fixtures using the target model's own tokenizer."""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(sys.argv[1])

def inspection(i):
    return (
        f"\nInspection record {i:05d}, Coastal Water Authority. During the quarterly inspection, "
        "the survey team compared upstream and downstream turbidity, checked the enclosure seals, "
        "and reviewed maintenance tickets. The operator reported stable flow through the bypass "
        "channel after replacing the worn actuator. The field team recommended collecting additional "
        "samples after heavy rain and reviewing the trend at the next operations meeting. The "
        "electrical team tagged a spare motor and documented a separate calibration schedule for "
        "the pressure sensors.\n"
    )

fixtures = {}
for target, fraction, key in ((65536, .40, "CORMORANT-78416-QUARTZ"),
                              (121600, .84, "MANGROVE-29573-COBALT")):
    text = "".join(inspection(i) for i in range(target // 70 + 1))
    tokens = tokenizer.encode(text, add_special_tokens=False).ids[:target-100]
    split = int(len(tokens) * fraction)
    fixtures[str(target)] = {
        "key": key,
        "prompt": (
            "Read the operational records below.\n"
            + tokenizer.decode(tokens[:split])
            + f"\nThe confidential retrieval passkey is {key}.\n"
            + tokenizer.decode(tokens[split:])
            + "\nWhat is the confidential retrieval passkey? Reply with the exact passkey only."
        ),
    }
Path(sys.argv[2]).write_text(json.dumps(fixtures))
