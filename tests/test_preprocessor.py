import unittest
import numpy as np
import cv2
from src.data.preprocessor import CandlingPreprocessor


class TestCandlingPreprocessor(unittest.TestCase):
    def test_preprocessor_resizing(self):
        preprocessor = CandlingPreprocessor(target_size=(256, 256))
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        processed = preprocessor.process(dummy_img)
        self.assertEqual(processed.shape, (256, 256, 3))

    def test_preprocessor_green_channel_extraction(self):
        preprocessor = CandlingPreprocessor(target_size=(640, 640), extract_green=True)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_img[:, :, 1] = 150
        g_channel = preprocessor.extract_g_channel(dummy_img, is_bgr=True)
        self.assertEqual(g_channel.shape, (100, 100))
        self.assertTrue(np.all(g_channel == 150))

    def test_preprocessor_clahe(self):
        preprocessor = CandlingPreprocessor(apply_clahe=True)
        gray = np.random.randint(50, 150, (100, 100), dtype=np.uint8)
        enhanced = preprocessor.apply_clahe_enhancement(gray)
        self.assertEqual(enhanced.shape, (100, 100))
        self.assertEqual(enhanced.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
