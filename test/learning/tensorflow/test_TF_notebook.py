#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/11
# @Time      : 17:57
import unittest

import numpy as np

from snorkel.annotations import FeatureAnnotator
from snorkel.learning.tensorflow import LogisticRegression, SparseLogisticRegression
from test.learning.tensorflow.tensorflow_test_base import TensorFlowTestBase

__author__ = 'zhnlk'


class TFNoiseAwareModel(TensorFlowTestBase):

    def _train_logistic_regression(self):

        model = LogisticRegression(cardinality=self.Tweet.cardinality)
        model.train(F_train.todense(), train_marginals)

        # train sparse logistic regression
        model = SparseLogisticRegression(cardinality=Tweet.cardinality)
        model.train(F_train, train_marginals, n_epochs=50, print_freq=10)


        acc = model.score(F_test, test_labels)
        print(acc)
        assert acc > 0.6

        # test with batch size s.t. N % batch_size == 1...
        model.score(F_test, test_labels, batch_size=9)

    def _train_basic_LSTM(self):
        sess


if __name__ == '__main__':
    unittest.main()
