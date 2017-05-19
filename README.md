# Regularisation

Regularisation is the method to reduce the overfitting of the network or model.
There are some approaches to reduce overfitting. They are 
1) Increasing the amount of training data.
2) Another approach is to reduce the size of the network. However, large networks have the potential to be more powerful than small networks, and so this is an option we'd only adopt reluctantly.

There are other techniques which can reduce overfitting, even when we have a fixed network and fixed training data. These are known as regularization techniques. 

## Regularisation Techniques:

1) L2 Regularisation.
2) L1 Regularisation.

## L2 Regularisation

This is the most commonly used regularisation techinques, sometimes also called as **weight decay**.<br />

### Regularisation of loss function :

Let us discuss how cost is regularised using this L2 Regularisation. We know that cross_entropy cost is 

<code>C = −1/n∑[y<sub>j</sub>lna<sub>j</sub> + (1−y<sub>j</sub>)ln(1−a<sub>j</sub>)]</code>

The idea behind L2 regularization is to add an extra term to the cost function, a term called the regularization term.
So after adding regularisation term the cross_entropy cost is given as

<code>C=−1/n∑[y<sub>j</sub>lna<sub>j</sub> + (1−y<sub>j</sub>)ln(1−a<sub>j</sub>)] + λ/2n∑w<sup>2</sup></code>

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

## L1 Regularisation

This is the another commonly used regularisation techinques.  Of course, the L1 regularization term isn't the same as the L2 regularization term, and so we shouldn't expect to get exactly the same behaviour. Let's try to understand how the behaviour of a network trained using L1 regularization differs from a network trained using L2 regularization.<br />

### Regularisation of loss function :

In this approach we modify the unregularized cost function by adding the sum of the absolute values of the weights:

<code>C = C<sub>0</sub> + λ/n∑|w|</code>

### Regularisation of Weights :

Now if we look at the partial derivatives of the cost function.

<code> ∂C/∂w = ∂C<sub>0</sub>/∂w + λ/n * sgn(w) </code>

where sgn(w) is the sign of w, that is, +1 if w is positive, and −1 if w is negative. Using this expression, we can easily modify backpropagation to do stochastic gradient descent using L1 regularization. The resulting update rule for an L1 regularized network is

<code>w → w′ = w − η(λ/n) * sgn(w) − η(∂C<sub>0</sub>/∂w</code>

<code>w → w′ = w(1 − η(λ/n)) − η(∂C<sub>0</sub>/∂w</code>

In both L2 and L1 techniques the effect of regularization is to shrink the weights. This accords with our intuition that both kinds of regularization penalize large weights. But the way the weights shrink is different. In L1 regularization, the weights shrink by a constant amount toward 0. In L2 regularization, the weights shrink by an amount which is proportional w. And so when a particular weight has a large magnitude, |w|, L1 regularization shrinks the weight much less than L2 regularization does. By contrast, when |w| is small, L1 regularization shrinks the weight much more than L2 regularization. The net result is that L1 regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero.

## Regularisation in Tensorflow.

### L2 Regularisation :

Let us discuss how to apply L2 regularisation in tensorflow code. The following is the unregularised cost function.

<code> cost = tf.reduce_mean(tf.square(actual - pred)) </code>

so to apply regularisation we have to add regularised parameter with sum of square of weights to actual cost function.

<code>regulariser = tf.nn.l2_loss(weights)</code>

The above tensorflow function is nothing but 

<code>regulariser  = (weights ** 2) / 2</code>

So finally our cost function will be 

<code>cost = tf.reduce_mean(cost + beta * regulariser)</code>

where beta is regularisation constant.

similarly if we have 2 hidden layers and we want to regularise cost based in two weights then 

<code> regulariser = tf.nn.l2_loss(weight1) + tf.nn.l2_loss(weight2)</code><br />
<code>cost = tf.reduce_mean(cost + beta * regulariser)</code>

So when we apply gradient then weight will regularised according to the regularisation parameter.

### L1 Regularisation :

Let us discuss how to apply L1 regularisation in tensorflow code.

<code> cost = tf.reduce_mean(tf.square(actual - pred)) </code>

so to apply regularisation we have to add regularised parameter with sum of absolute values of weights to actual cost function.

<code>regulariser = tf.reduce_sum(tf.abs(weight))</code>

The above tensorflow function is nothing but 

<code>regulariser  = sum(absolute(weight))</code>

So finally our cost function will be 

<code>cost = tf.reduce_mean(cost + beta * regulariser)</code>

where beta is regularisation constant.

similarly if we have 2 hidden layers and we want to regularise cost based in two weights then 

<code> regulariser = tf.reduce_sum(tf.abs(weight1)) + tf.reduce_sum(tf.abs(weight2))</code><br />
<code>cost = tf.reduce_mean(cost + beta * regulariser)</code>

So when we apply gradient then weight will regularised according to the regularisation parameter.



# References :

[Regularisation methods by Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap3.html)


















