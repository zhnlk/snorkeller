#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/11
# @Time      : 20:51
import unittest

from snorkel.learning.tensorflow import SparseLogisticRegression
from test.learning.tensorflow.tensorflow_test_base import TensorFlowTestBase

__author__ = 'zhnlk'


class TestLogisticRegression(TensorFlowTestBase):

    def test_train_sparse_logistic_regression(self):
        model = SparseLogisticRegression(cardinality=self.Tweet.candinality)
        model.train(self.F_train, self.train_marginals, n_epochs=50, print_freq=10)

        acc = model.train(self.F_test, self.test_labels)
        print(acc)

        assert acc > 0.6

        # Test with batch size s.t. N % batch_size == 1...
        model.score(self.F_test, self.test_labels, batch_size=9)


if __name__ == '__main__':
    unittest.main()
