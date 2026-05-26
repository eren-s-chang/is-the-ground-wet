import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from inference.pavement_classifier import PavementWetDryClassifier, crop_roi


class PavementClassifierTests(unittest.TestCase):
    def test_bottom_roi_crop_shape(self):
        image = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
        crop, box = crop_roi(image, roi=(0.25, 0.5, 0.75, 1.0))
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(box, (50, 50, 150, 100))

    def test_classifier_initializes_with_heuristic_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "missing.pt"
            classifier = PavementWetDryClassifier(model_path=str(model_path), fallback_mode="heuristic")
            self.assertIsNone(classifier.model)
            result = classifier.predict(Image.fromarray(np.full((120, 120, 3), 220, dtype=np.uint8)))
            self.assertIn(result.label, {"wet", "dry"})
            self.assertGreaterEqual(result.confidence, 0)
            self.assertLessEqual(result.confidence, 1)


if __name__ == "__main__":
    unittest.main()
