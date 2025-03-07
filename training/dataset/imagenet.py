from pathlib import Path

import datasets


class ImageNet(datasets.GeneratorBasedBuilder):
    """ImageNet Class Name Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "class_id": datasets.Value("string"),
                "class_name": datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager):
        data_path = Path('./classnames.txt')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                },
            ),
        ]

    def _generate_examples(self, data_path):
        counter = 0
        with open(data_path, 'r') as data:
            class_data = data.read().splitlines()
        for c in class_data:
            c_id, c_name = c.split(' ', 1)
            yield counter, {
                "class_id": c_id,
                "class_name": c_name,
            }
            counter += 1
