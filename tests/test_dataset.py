from PIL import Image

from mpb_latex_ocr.data.dataset import FormulaImageTransform


def test_handwriting_augmentation_profile_returns_fixed_tensor_shape():
    transform = FormulaImageTransform(
        height=64,
        width=128,
        augment=True,
        augmentation_profile="handwriting",
        augmentation_strength=0.5,
    )
    image = Image.new("L", (80, 30), color=255)

    tensor = transform(image)

    assert tuple(tensor.shape) == (1, 64, 128)
