import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu

def multiply(x):
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(positive, negative, margin):
    """
        Calculate Average Memory Loss Function
        positive - positive cosine similarity
        negative - negative cosine similarity
        margin
    """
    assert(positive.size() == negative.size())
    dist_hinge = torch.clamp(negative - positive + margin, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss

"""
Softmax Temperature -
    + Assume we have K elements at distance x. One element is at distance x+a
    + e^tm(x+a) / K*e^tm*x + e^tm(x+a) = e^tm*a / K + e^tm*a
    + For 20% probability, e^tm*a = 0.2K -> tm = ln(0.2 K)/a
"""

class Memory(nn.Module):
    def __init__(self, memory_size, key_dim, top_k = 256, inverse_temp = 40, age_noise=8.0, margin = 0.1):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.top_k = min(top_k, memory_size)
        self.softmax_temperature = max(1.0, math.log(0.2 * top_k) / inverse_temp)
        self.age_noise = age_noise
        self.margin = margin

        # Parameters
        self.build()
        self.query_proj = nn.Linear(key_dim, key_dim)

    def build(self):
        self.keys = F.normalize(random_uniform((self.memory_size, self.key_dim), -0.001, 0.001, cuda=True), dim=1)
        self.keys_var = ag.Variable(self.keys, requires_grad=False)
        self.values = torch.zeros(self.memory_size, 1).long().cuda()
        self.age = torch.zeros(self.memory_size, 1).cuda()

    def predict(self, x):
        batch_size, dims = x.size()
        query = F.normalize(self.query_proj(x), dim=1)

        # Find the k-nearest neighbors of the query
        scores = torch.matmul(query, torch.t(self.keys_var))
        cosine_similarity, topk_indices_var = torch.topk(scores, self.top_k, dim=1)

        # softmax of cosine similarities - embedding
        softmax_score = F.softmax(self.softmax_temperature * cosine_similarity)

        # retrive memory values - prediction
        y_hat_indices = topk_indices_var.data[:, 0]
        y_hat = self.values[y_hat_indices]

        return y_hat, softmax_score

    def query(self, x, y, predict=False):
        """
        Compute the nearest neighbor of the input queries.

        Arguments:
            x: A normalized matrix of queries of size (batch_size x key_dim)
            y: A matrix of correct labels (batch_size x 1)
        Returns:
            y_hat, A (batch-size x 1) matrix 
		        - the nearest neighbor to the query in memory_size
            softmax_score, A (batch_size x 1) matrix 
		        - A normalized score measuring the similarity between query and nearest neighbor
            loss - average loss for memory module
        """
        batch_size, dims = x.size()
        query = F.normalize(self.query_proj(x), dim=1)
        #query = F.normalize(torch.matmul(x, self.query_proj), dim=1)

        # Find the k-nearest neighbors of the query
        scores = torch.matmul(query, torch.t(self.keys_var))
        cosine_similarity, topk_indices_var = torch.topk(scores, self.top_k, dim=1)

        # softmax of cosine similarities - embedding
        softmax_score = F.softmax(self.softmax_temperature * cosine_similarity)

        # retrive memory values - prediction
        topk_indices = topk_indices_var.detach().data
        y_hat_indices = topk_indices[:, 0]
        y_hat = self.values[y_hat_indices]

        loss = None
        if not predict:
            # Loss Function
            # topk_indices = (batch_size x topk)
            # topk_values =  (batch_size x topk x value_size)

            # collect the memory values corresponding to the topk scores
            batch_size, topk_size = topk_indices.size()
            flat_topk = flatten(topk_indices)
            flat_topk_values = self.values[topk_indices]
            topk_values = flat_topk_values.resize_(batch_size, topk_size)

            correct_mask = torch.eq(topk_values, torch.unsqueeze(y.data, dim=1)).float()
            correct_mask_var = ag.Variable(correct_mask, requires_grad=False)

            pos_score, pos_idx = torch.topk(torch.mul(cosine_similarity, correct_mask_var), 1, dim=1)
            neg_score, neg_idx = torch.topk(torch.mul(cosine_similarity, 1-correct_mask_var), 1, dim=1)

            # zero-out correct scores if there are no correct values in topk values
            mask = 1.0 - torch.eq(torch.sum(correct_mask_var, dim=1), 0.0).float()
            pos_score = torch.mul(pos_score, torch.unsqueeze(mask, dim=1))

            #print(pos_score, neg_score)
            loss = MemoryLoss(pos_score, neg_score, self.margin)

        # Update memory
        self.update(query, y, y_hat, y_hat_indices) 

        return y_hat, softmax_score, loss

    def update(self, query, y, y_hat, y_hat_indices):
        batch_size, dims = query.size()

        # 1) Untouched: Increment memory by 1
        self.age += 1

        # Divide batch by correctness
        result = torch.squeeze(torch.eq(y_hat, torch.unsqueeze(y.data, dim=1))).float()
        incorrect_examples = torch.squeeze(torch.nonzero(1-result))
        correct_examples = torch.squeeze(torch.nonzero(result))

        incorrect = len(incorrect_examples.size()) > 0
        correct = len(correct_examples.size()) > 0

        # 2) Correct: if V[n1] = v
        # Update Key k[n1] <- normalize(q + K[n1]), Reset Age A[n1] <- 0
        if correct:
            correct_indices = y_hat_indices[correct_examples]
            correct_keys = self.keys[correct_indices]
            correct_query = query.data[correct_examples]

            new_correct_keys = F.normalize(correct_keys + correct_query, dim=1)
            self.keys[correct_indices] = new_correct_keys
            self.age[correct_indices] = 0

        # 3) Incorrect: if V[n1] != v
        # Select item with oldest age, Add random offset - n' = argmax_i(A[i]) + r_i 
        # K[n'] <- q, V[n'] <- v, A[n'] <- 0
        if incorrect:
            incorrect_size = incorrect_examples.size()[0]
            incorrect_query = query.data[incorrect_examples]
            incorrect_values = y.data[incorrect_examples]

            age_with_noise = self.age + random_uniform((self.memory_size, 1), -self.age_noise, self.age_noise, cuda=True)
            topk_values, topk_indices = torch.topk(age_with_noise, incorrect_size, dim=0)
            oldest_indices = torch.squeeze(topk_indices)

            self.keys[oldest_indices] = incorrect_query
            self.values[oldest_indices] = incorrect_values
            self.age[oldest_indices] = 0
