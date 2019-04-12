#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/12
# @Time      : 16:10

__author__ = 'zhnlk'

import numpy as np


def test_load():
    # np.save('123', np.array([[1, 2, 3], [4, 5, 6]]))
    # a = np.load('123.npy')
    # print(a)
    train_labels = np.load('crowdsourcing_train_labels.npy')
    print(train_labels)
