import unittest
from source import preprocessing_audio
from source import data_loaders

class TestPreprocessing(unittest.TestCase):

    def test_transformer(self):
        test = preprocessing_audio.simple_transformer(
            mp3path = './data/raw/CAL500_32kps/2pac-trapped.mp3',
            savedirectory='./data/test/',
            filename='test-file',
            sample_rate=32000,
            seconds=30,
            offset=15.0
        )

        self.assertTrue(test)

class TestDataloaders(unittest.TestCase):

    def test_stft_load(self):
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loaders.load_section_level_stft(preprocessing='stft', dev=True)
        self.assertEqual(train_features.shape[0], train_labels.shape[0])
        self.assertEqual(valid_features.shape[0], valid_labels.shape[0])
        self.assertEqual(test_features.shape[0], test_labels.shape[0])

    def test_chroma_load(self):
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loaders.load_section_level_stft(preprocessing='chroma', dev=True)
        self.assertEqual(train_features.shape[0], train_labels.shape[0])
        self.assertEqual(valid_features.shape[0], valid_labels.shape[0])
        self.assertEqual(test_features.shape[0], test_labels.shape[0])

    def test_mfcc_load(self):
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loaders.load_section_level_stft(preprocessing='mfcc', dev=True)
        self.assertEqual(train_features.shape[0], train_labels.shape[0])
        self.assertEqual(valid_features.shape[0], valid_labels.shape[0])
        self.assertEqual(test_features.shape[0], test_labels.shape[0])

    def test_wave_load(self):
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data_loaders.load_section_level_stft(preprocessing='wave', dev=True)
        self.assertEqual(train_features.shape[0], train_labels.shape[0])
        self.assertEqual(valid_features.shape[0], valid_labels.shape[0])
        self.assertEqual(test_features.shape[0], test_labels.shape[0])


if __name__ == "__main__":
    unittest.main()