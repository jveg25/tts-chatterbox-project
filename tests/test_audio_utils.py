import unittest
import numpy as np
import os
from src.backend.audio_utils import stitch_audio

class TestAudioUtils(unittest.TestCase):
    def test_stitch_audio(self):
        w1 = np.array([0.1, 0.2, 0.3])
        w2 = np.array([0.4, 0.5])
        stitched = stitch_audio([w1, w2])
        self.assertEqual(len(stitched), 5)
        np.testing.assert_array_equal(stitched, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    def test_stitch_audio_empty(self):
        self.assertEqual(len(stitch_audio([])), 0)

if __name__ == "__main__":
    unittest.main()
