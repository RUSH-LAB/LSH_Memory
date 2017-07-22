import torch
import torch.autograd as ag
import torch.nn as nn
import numpy as np

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    return (high - low) * x + low

def normalize(matrix, dim):
    """ Normalize Matrix - Unit Norm """
    total = torch.sum(matrix, dim)
	return matrix / total

# TODO Memory Loss Function

"""
Softmax Temperature -
    + Assume we have K elements at distance x. One element is at distance x+a
    + e^tm(x+a) / K*e^tm*x + e^tm(x+a) = e^tm*a / K + e^tm*a
    + For 20% probability, e^tm*a = 0.2K -> tm = ln(0.2 K)/a
"""

class Memory(nn.Module):
    def __init__(self, memory_size, key_dim, top_k = 256, inverse_temp = 40, age_noise=8.0, margin = 0.1):
        #Initialize normalized key matrix
        self.keys = nn.Parameter(normalize(torch.randn(memory_size, key_dim), dim=1))
        self.value = nn.Parameter(torch.zeros(memory_size, 1))
        self.age = nn.Parameter(torch.zeros(memory_size, 1))

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.top_k = min(top_k, memory_size)
        self.softmax_temp = max(1.0, torch.log(0.2 * topk) / inverse_temp)
        self.age_noise = age_noise
        self.margin = margin

    def forward(self, input, labels):
        """
        Compute the nearest neighbor of the input queries.

        Arguments:
            input: A normalized matrix of queries of size (batch_size x key_dim)
        Returns:
            result, A (batch-size x 1) matrix 
		- the nearest neighbor to the query in memory_size
            softmax_score, A (batch_size x 1) matrix 
		- A normalized score measuring the similarity between query and nearest neighbor
        """
        normalized_input = normalize(input, dim=1)

        # Find the k-nearest neighbors of the query
        scores = torch.matmul(normalized_input, torch.t(self.keys.data))
        values, indices = torch.topk(scores, self.top_k, dim = 1)
        result = self.value[indices[:, 0]]

        # Calculate similarity values
        cosine_similarity = torch.dot(normalized_input, torch.t(self.keys[indices, :]))
        softmax_score = nn.Softmax(self.inverse_temperature * cosine_similarity)

        # Update memory
        self.update(normalize_input, labels, indices) 

        return result, softmax_score

    def update(query, y, y_hat):
        # 1) Untouched: Increment memory by 1
        self.age = torch.add(self.age, 1)

        # indices
        correct = torch.eq(y, y_hat)
        correct_indices = torch.nonzero(correct)
        incorrect_indices = torch.nonzero(1-correct)

        # 2) Correct: if V[n1] = v
        # Update Key k[n1] <- normalize(q + K[n1]), Reset Age A[n1] <- 0
        correct_values = self.value[correct_indices]
        correct_query = query[correct_indices]
        update_correct_values = normalize(correct_values + correct_query, dim=1)
        self.age[correct_indices] = 0

        # 3) Incorrect: if V[n1] != v
        # Select item with oldest age, Add random offset - n' = argmax_i(A[i]) + r_i 
        # K[n'] <- q, V[n'] <- v, A[n'] <- 0
        age_with_noise = self.age + random_uniform((self.memory_size, 1), -self.age_noise, self.age_noise)
        valuse, oldest_indices = torch.topk(age_with_noise, self.batch_size, dim = 0)
        self.keys[oldest_indices] = query
        self.values[oldest_indices] = y
        self.age[oldest_indices] = 0

