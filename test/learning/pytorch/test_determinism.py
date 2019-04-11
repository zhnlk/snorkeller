import unittest

import torch

from .pytorch_test_base import PyTorchTestBase
from snorkel.learning.pytorch import LSTM


class TestDeterminism(PyTorchTestBase):

    def test_lstm_determinism(self):
        train_kwargs = {
            'lr': 0.01,
            'embedding_dim': 50,
            'hidden_dim': 50,
            'n_epochs': 2,
            'dropout': 0.25,
            'num_layers': 1,
            'bidirectional': False
        }

        lstm1 = LSTM()
        lstm1.train(self.train_cands, self.train_marginals, **train_kwargs)

        lstm2 = LSTM()
        lstm2.train(self.train_cands, self.train_marginals, **train_kwargs)

        self.assertTrue(torch.sum(torch.abs(
            lstm1.output_layer.weight.data - lstm2.output_layer.weight.data
        )) < 1e-4)


if __name__ == '__main__':
    unittest.main()
