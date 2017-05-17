# Regularisation

Regularisation is the method to reduce the overfitting of the network or model.
There are some approaches to reduce overfitting. They are 
1) Increasing the amount of training data.
2) Another approach is to reduce the size of the network. However, large networks have the potential to be more powerful than small networks, and so this is an option we'd only adopt reluctantly.

There are other techniques which can reduce overfitting, even when we have a fixed network and fixed training data. These are known as regularization techniques. 

## Regularisation Techniques:

1) L2 Regularisation.
2) L1 Regularisation.
3) Bias Regularisation.

## L2 Regularisation

This is the most commonly used regularisation techinques, sometimes also called as **weight decay**.<br />

### Regularisation of loss function :

Let us discuss how cost is regularised using this L2 Regularisation. We know that cross_entropy cost is 

<code>C = −1/n∑[y<sub>j</sub>lna<sub>j</sub>+(1−y<sub>j</sub>)ln(1−a<sub>j</sub>)]</code>

The idea behind L2 regularization is to add an extra term to the cost function, a term called the regularization term.
So after adding regularisation term the cross_entropy cost is given as

<code>C=−1/n∑[y<sub>j</sub>lna<sub>j</sub>+(1−y<sub>j</sub>)ln(1−a<sub>j</sub>)] + λ/2n∑w<sup>2</sup></code>

The first term is just the usual expression for the cross-entropy. But we've added a second term, namely the sum of the squares of all the weights in the network. This is scaled by a factor λ/2n, where λ>0 is known as the regularization parameter, and n is, as usual, the size of our training set.

It's possible to regularize other cost functions, such as the quadratic cost. This can be done in a similar way:

<code>C = 1/2n∑‖y−a<sup>L</sup>‖<sup>2</sup> + λ/2n∑w<sup>2</sup></code>

So in both cases we can write the regularized cost function as

<code>C = C<sub>0</sub> + λ/2n∑w<sup>2</sup></code>
where C<sub>0</sub> is the original, unregularized cost function.

### Regularisation of Weights and bias :

Let us discuss about gradient descent learning algorithm in a regularized neural network. In particular, we need to know how to compute the partial derivatives ∂C/∂w and ∂C/∂b for all the weights and biases in the network.

<code>∂C/∂w = ∂C<sub>0</sub>/∂w + (λ/n)w</code><br /><code>∂C/∂b = ∂C<sub>0</sub>/∂b</code>

And so we see that it's easy to compute the gradient of the regularized cost function: just use backpropagation, as usual, and then add (λ/n)w to the partial derivative of all the weight terms. The partial derivatives with respect to the biases are unchanged, and so the gradient descent learning rule for the biases doesn't change from the usual rule:

<code>b → b−η(∂C<sub>0</sub>/∂b)</code>

The learning rule for the weights becomes:

<code>w → w − η(∂C<sub>0</sub>/∂w) − η(λ/n)w</code><br /><code> = (1 − η(λ/n))w − η(∂C<sub>0</sub>/∂w)</code>

This is exactly the same as the usual gradient descent learning rule, except we first rescale the weight w by a factor 1 − η(λ/n). This rescaling is sometimes referred to as weight decay, since it makes the weights smaller. At first glance it looks as though this means the weights are being driven unstoppably toward zero. But that's not right, since the other term may lead the weights to increase, if so doing causes a decrease in the unregularized cost function.

### Regularisation of Weights and bias in stochastic gradient descent:

Let us see the how weights and bias are regularised in stochastic gradient descent

<code> w → (1 − η(λ/n))w − η/m∑(∂C<sub>x</sub>/∂w)</code>

<code>b → b−η/m(∂C<sub>x</sub>/∂b)</code>

where the sum is over training examples x in the mini-batch.



















