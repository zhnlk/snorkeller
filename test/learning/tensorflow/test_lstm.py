# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/11
# @Time      : 20:55
import unittest

from snorkel.learning.tensorflow import TextRNN
from .tensorflow_test_base import TensorFlowTestBase

__author__ = 'zhnlk'


class TestLSTM(TensorFlowTestBase):
    def test_lstm_architectures(self):
        pass

    def test_lstm_with_dev_set(self):
        train_kwargs = {
            'dim': 100,
            'lr': 0.001,
            'n_epochs': 25,
            'dropout': 0.2,
            'print_freq': 5
        }
        lstm = TextRNN(seed=123, cardinality=self.Tweet.cardinality)
        lstm.train(self.train_tweets, self.train_marginals, X_dev=self.test_tweets, Y_dev=self.test_labels,
                   **train_kwargs)
        acc = lstm.score(self.test_tweets, self.test_labels)
        print(acc)
        assert acc > 0.60

        # Test with batch size s.t. N % batch_size == 1...
        lstm.score(self.test_tweets, self.test_labels, batch_size=9)


if __name__ == '__main__':
    unittest.main()
