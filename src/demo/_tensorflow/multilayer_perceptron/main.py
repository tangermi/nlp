# -*- coding:utf-8 -*-
from train import MLP
from evaluation import Evaluator

if __name__ == '__main__':
    mlp = MLP()
    evaluator = Evaluator()
    mlp.train()
    evaluator.model = mlp.model
    evaluator.batch_size = mlp.batch_size
    evaluator.evaluate()
