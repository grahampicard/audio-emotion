import unittest
from source import preprocessing_audio
from source import data_loaders

class TestPreprocessing(unittest.TestCase):

    def test_transformer(self):
        test = preprocessing_audio.simple_transformer(
            mp3path = './data/raw/CAL500_32kps/2pac-trapped.mp3',
            savedirectory='./data/test/',
            filename='test-file',
            transforms=['stft'],
            sample_rate=32000,
            seconds=30,
            offset=15.0
        )

        self.assertTrue(test)

    def test_generic_loader(self):
        test = data_loaders.generic_loader()

        self.assertEqual(test[0].shape[0], test[1].shape[0])
        self.assertEqual(test[2].shape[0], test[3].shape[0])
        self.assertEqual(test[4].shape[0], test[5].shape[0])


if __name__ == "__main__":
    unittest.main()