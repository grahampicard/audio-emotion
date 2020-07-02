import unittest
from source import preprocessing_audio
from source import data_loaders

class TestPreprocessing(unittest.TestCase):

    def test_transformer(self):
        test = preprocessing_audio.simple_stft_transform(
            mp3path = './data/raw/CAL500_32kps/2pac-trapped.mp3',
            savedirectory='./data/test/',
            filename='test-file',
            transforms=['stft'],
            sample_rate=32000,
            seconds=30,
            offset=15.0
        )

        self.assertTrue(test)

class TestLoaders(unittest.TestCase):

    def test_cal500_loader(self):
        train_features, train_labels, \
        test_features, test_labels = data_loaders.load_stft_data()        

        self.assertEqual(res[0][0], 395)
        self.assertEqual(res[0][2] = 1025)
        self.assertEqual(res[0][2] = 1025)

if __name__ == "__main__":
    unittest.main()