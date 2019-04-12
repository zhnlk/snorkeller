#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date      : 2019/4/11
# @Time      : 20:43
import os
import unittest

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from snorkel.annotations import FeatureAnnotator, load_marginals
from snorkel.models import candidate_subclass

__author__ = 'zhnlk'


class TensorFlowTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        We'll start by testing the `textRNN` model on a categorical problem from `tutorials/crowdsourcing`.
        In particular we'll test for (a) basic performance and (b) proper construction / re-construction of
        the TF computation graph both after (i) repeated notebook calls, and (ii) with `GridSearch` in particular.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        snorkel_engine = create_engine(os.path.join('sqlite:///' + dir_path, 'crowdsourcing.db'))
        SnorkelSession = sessionmaker(bind=snorkel_engine)
        cls.session = SnorkelSession()

        cls.Tweet = candidate_subclass('Tweet', ['tweet'], cardinality=5)

        cls.train_tweets = cls.session.query(cls.Tweet).filter(cls.Tweet.split == 0).order_by(cls.Tweet.id).all()
        # cls.train_labels = np.load('test/learning/tensorflow/crowdsourcing_train_labels.npy')
        cls.train_labels = np.load('crowdsourcing_train_labels.npy')
        cls.train_marginals = load_marginals(cls.session, cls.train_tweets, split=0)
        # test/learning/tensorflow/crowdsourcing_test_labels.npy
        cls.test_tweets = cls.session.query(cls.Tweet).filter(cls.Tweet.split == 1).order_by(cls.Tweet.id).all()
        # cls.test_labels = np.load('test/learning/tensorflow/crowdsourcing_test_labels.npy')
        cls.test_labels = np.load('crowdsourcing_test_labels.npy')

        # simple unigrame featurizer
        def get_unigram_tweet_features(c):
            for w in c.tweet.text.split():
                yield w, 1

        # construct feature matrix
        cls.featurizer = FeatureAnnotator(get_unigram_tweet_features)
        cls.F_train = cls.featurizer.apply(split=0)

        cls.F_test = cls.featurizer.apply_existing(split=1)

    @classmethod
    def tearDownClass(cls):
        cls.session.close()
