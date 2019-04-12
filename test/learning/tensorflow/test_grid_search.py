#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/11
# @Time      : 20:56
import unittest

from snorkel.learning import GridSearch
from snorkel.learning.tensorflow import TextRNN
from test.learning.tensorflow.tensorflow_test_base import TensorFlowTestBase

__author__ = 'zhnlk'


class TestGridSearch(TensorFlowTestBase):

    def test_searching_over_learning_rate(self):
        param_ranges = {
            'lr': [1e-3, 1e-4],
            'dim': [50, 100]
        }
        model_class_params = {
            'seed': 123,
            'cardinality': self.Tweet.cardinality
        }
        model_hyperparams = {
            'dim': 100,
            'n_epochs': 20,
            'dropout': 0.1,
            'print_freq': 10
        }
        searcher = GridSearch(TextRNN, param_ranges, self.train_tweets, self.train_marginals,
                              model_class_params=model_class_params,
                              model_hyperparams=model_hyperparams)
        # use test set here ,just for testing
        lstm, run_stats = searcher.fit(self.test_tweets, self.test_labels)

        acc = lstm.score(self.test_tweets, self.test_labels)
        print(acc)
        assert acc > 0.6


if __name__ == '__main__':
    unittest.main()
