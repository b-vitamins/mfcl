from pathlib import Path

from PIL import Image

from mfcl.data.imagenet1k import ImageListDataset


def test_imagelist_dataset_allows_repeated_reads(tmp_path):
    root = Path(tmp_path)
    image_path = root / "img.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(image_path)

    file_list = root / "files.txt"
    file_list.write_text(f"{image_path}\n", encoding="utf-8")

    dataset = ImageListDataset(str(root), str(file_list))

    def assert_file_readable() -> None:
        with Image.open(image_path) as handle:
            handle.load()

    for _ in range(2):
        sample, _ = dataset[0]
        assert sample["img"].size == (4, 4)
        assert_file_readable()
