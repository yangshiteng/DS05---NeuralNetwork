# 1. What are Neural Networks?

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

![image](https://user-images.githubusercontent.com/60442877/149044081-6a89d90a-4854-456b-8ecd-0e731ff38d2f.png)

Neural networks rely on training data to learn and improve their accuracy over time. However, once these learning algorithms are fine-tuned for accuracy, they are powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts. One of the most well-known neural networks is Google’s search algorithm.

# 2. How do Neural Networks work?

Think of each individual node as its own linear regression model, composed of input data, weights, a bias (or threshold), and an output. The formula would look something like this:

![image](https://user-images.githubusercontent.com/60442877/149044693-78fbdf05-5ad0-4145-886d-4cdc081c0eb1.png)

![image](https://user-images.githubusercontent.com/60442877/149044722-4cc03a2c-2cda-440e-8c20-4f6a71449b7c.png)

Once an input layer is determined, weights are assigned. These weights help determine the importance of any given variable, with larger ones contributing more significantly to the output compared to other inputs. All inputs are then multiplied by their respective weights and then summed. Afterward, the output is passed through an activation function, which determines the output. If that output exceeds a given threshold, it “fires” (or activates) the node, passing data to the next layer in the network. This results in the output of one node becoming in the input of the next node. This process of passing data from one layer to the next layer defines this neural network as a feedforward network.

Let’s break down what one single node might look like using binary values. We can apply this concept to a more tangible example, like whether you should go surfing (Yes: 1, No: 0). The decision to go or not to go is our predicted outcome, or y-hat. Let’s assume that there are three factors influencing your decision-making:

- Are the waves good? (Yes: 1, No: 0)
- Is the line-up empty? (Yes: 1, No: 0)
- Has there been a recent shark attack? (Yes: 0, No: 1)

Then, let’s assume the following, giving us the following inputs:

- X1 = 1, since the waves are pumping
- X2 = 0, since the crowds are out
- X3 = 1, since there hasn’t been a recent shark attack

Now, we need to assign some weights to determine importance. Larger weights signify that particular variables are of greater importance to the decision or outcome.

- W1 = 5, since large swells don’t come around often
- W2 = 2, since you’re used to the crowds
- W3 = 4, since you have a fear of sharks

Finally, we’ll also assume a threshold value of 3, which would translate to a bias value of –3. With all the various inputs, we can start to plug in values into the formula to get the desired output.

Y-hat = (1*5) + (0*2) + (1*4) – 3 = 6

If we use the activation function from the beginning of this section, we can determine that the output of this node would be 1, since 6 is greater than 0. In this instance, you would go surfing; but if we adjust the weights or the threshold, we can achieve different outcomes from the model. When we observe one decision, like in the above example, we can see how a neural network could make increasingly complex decisions depending on the output of previous decisions or layers.

In the example above, we used perceptrons to illustrate some of the mathematics at play here, but neural networks leverage sigmoid neurons, which are distinguished by having values between 0 and 1. Since neural networks behave similarly to decision trees, cascading data from one node to another, having x values between 0 and 1 will reduce the impact of any given change of a single variable on the output of any given node, and subsequently, the output of the neural network.

As we start to think about more practical use cases for neural networks, like image recognition or classification, we’ll leverage supervised learning, or labeled datasets, to train the algorithm. As we train the model, we’ll want to evaluate its accuracy using a cost (or loss) function. This is also commonly referred to as the mean squared error (MSE). In the equation below,

![image](https://user-images.githubusercontent.com/60442877/149045959-e797bba7-c14e-4452-aef8-48e68308c7a0.png)

Ultimately, the goal is to minimize our cost function to ensure correctness of fit for any given observation. As the model adjusts its weights and bias, it uses the cost function and reinforcement learning to reach the point of convergence, or the local minimum. The process in which the algorithm adjusts its weights is through gradient descent, allowing the model to determine the direction to take to reduce errors (or minimize the cost function). With each training example, the parameters of the model adjust to gradually converge at the minimum.  

![image](https://user-images.githubusercontent.com/60442877/149045980-e448b266-f193-449c-870c-c6873dbc2427.png)

Most deep neural networks are feedforward, meaning they flow in one direction only, from input to output. However, you can also train your model through backpropagation; that is, move in the opposite direction from output to input. Backpropagation allows us to calculate and attribute the error associated with each neuron, allowing us to adjust and fit the parameters of the model(s) appropriately.

# 3. Types of Neural Networks

Neural networks can be classified into different types, which are used for different purposes. While this isn’t a comprehensive list of types, the below would be representative of the most common types of neural networks that you’ll come across for its common use cases:

## 3.1 Perceptron

The perceptron is the oldest neural network, created by Frank Rosenblatt in 1958. It has a single neuron and is the simplest form of a neural network:

![image](https://user-images.githubusercontent.com/60442877/149046542-3852975a-d6b2-4597-8fd8-88ac04ed9914.png)

## 3.2 Feedforward Neural Networks (Multi-layer Perceptrons)

Feedforward neural networks, or multi-layer perceptrons (MLPs), are what we’ve primarily been focusing on within this article. They are comprised of an input layer, a hidden layer or layers, and an output layer. While these neural networks are also commonly referred to as MLPs, it’s important to note that they are actually comprised of sigmoid neurons, not perceptrons, as most real-world problems are nonlinear. Data usually is fed into these models to train them, and they are the foundation for computer vision, natural language processing, and other neural networks.

## 3.3 Convolutional Neural Networks (CNNs)

Convolutional neural networks (CNNs) are similar to feedforward networks, but they’re usually utilized for image recognition, pattern recognition, and/or computer vision. These networks harness principles from linear algebra, particularly matrix multiplication, to identify patterns within an image.

## 3.4 Recurrent Neural Networks (RNNs)

Recurrent neural networks (RNNs) are identified by their feedback loops. These learning algorithms are primarily leveraged when using time-series data to make predictions about future outcomes, such as stock market predictions or sales forecasting.


# 4. Neural Networks vs Deep Learning

Deep Learning and neural networks tend to be used interchangeably in conversation, which can be confusing. As a result, it’s worth noting that the “deep” in deep learning is just referring to the depth of layers in a neural network. A neural network that consists of more than three layers (or more than one hidden layer), which would be inclusive of the inputs and the output—can be considered a deep learning algorithm. A neural network that only has two or three layers is just a basic neural network.


# 5. Video Tutorial For Neural Netwworks (StatQuest ) 

## 5.1 https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1

- A Neural Network starts out with unknown parameter values that are estimated when we fit the Neural Network to a dataset using a method called Backpropagation
- When you build a Neural Network, you have to decide which activation function you want to use (for example, sigmoid function, Rectified Linear Unit (ReLU) function and softplus function)
- Sigmoid Function: ![image](https://user-images.githubusercontent.com/60442877/149124411-ab624a37-f77f-46e9-a76a-92ba61d6664e.png)
- ReLU Function: f(x) = max(0,x)
- Softplus Function: ![image](https://user-images.githubusercontent.com/60442877/149124656-084f0cf9-c7d8-4daf-908b-f183d63f9de0.png)


## 5.2 https://www.youtube.com/watch?v=wl1myxrtQHQ&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=2

## 5.3 https://www.youtube.com/watch?v=sDv4f4s2SB8&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=3

- gradient descent 
- step size = learning rate * derivative
- new parameter = old parameter - step size
- gradient descent stop when step size is very small (<= 0.001)
- even if the step size is still large, if therre have been more than the maximum number of steps (1000), gradient descent will stop
- When you have two or more derivatives of the same function, they are called a gradient
- Gradient Descent is very sensitive to the learning rate
- The good news is that in practice, a reasonable learning rate can be determined automatically by starting large and getting smaller with each step
- Step1: Take the derivative of the Loss Function for each parameter in it
- Step2: Pick random values for the parameters
- Step3: Plug the parameter values into the derivatives
- Step4: Calculate the step sizes: Step size = Slop * Learning rate
- Step5: Calculate the new parameter = old parameter - step size

Notes: When you have millions of data points, gradient descent can take a long time, so there is a thing called "Stochastic Gradient Descent" that uses a randomly selected subset of data at every step rather than the full dataset. This reduces the time spent calculating the derivatives of the loss function.

## 5.4 https://www.youtube.com/watch?v=IN2XmBhILt4&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=4

- Neural Network starts with identical activation function, but, using different weights and biases on the connections, it flips and stretches the activation functions into new shapes
- We fit Neural Networks to the dataset by backpropagation procedure
- Backpropagation starts with the last parameter and works its way backwards to estimate all of the other parameters

## 5.5 https://www.youtube.com/watch?v=iyn2zdALii8&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=5

- bias term frequently start at 0

## 5.6 https://www.youtube.com/watch?v=GKZoOHXGcLo&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=6

- We can use standard normal distribution to initialize weights
- Bias can ususally take 0 as initial value

## 5.7 https://www.youtube.com/watch?v=68BZ5f7P94E&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=7

- ReLU function may seem weird because it is not curvy

## 5.8 https://www.youtube.com/watch?v=83LYR-1IcjA&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=8

- We make the decision based on ArgMax or SoftMax
- We only use activation function if we want to pass the value to next layer, so, in the output layer, no need to use activation function

## 5.9 https://www.youtube.com/watch?v=KpKog-L9veg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=9

- We can't use Argmax in backpropagation
- In contrast, we can use Softmax in backpropagation
- ArgMax, just take max value as the final decision
- SoftMax function: ![image](https://user-images.githubusercontent.com/60442877/149148829-3471b1bb-4119-456b-b4dd-efeddb33668c.png)
- Unlike the ArgMax function, which can't be used in backpropagation since it has a derivative equal to 0 or its undefined, the derivative of the SoftMax function is not always 0 and we can use it for Gradient Descent
- So, we see why Neural Networks with multiple outputs often use SoftMax for training, then, use ArgMax, which has super easy to understand output, to classify new observation
- For ArgMax, we use sum of square residual to determine how well the Neural Network fit the data, however, when we use the SoftMax function, because the output values are predicted probabilities between 0 and 1, we often use something called "Cross Entropy" to determine how well the Neural Network fits the data
- ArgMax function, Loss function: Sum of Square Residuals, final result is either 1 or 0, and the values taken into ArgMax function is not probability
- SoftMax function, Loss function: Cross Entropy, final result is probability

## 5.10 https://www.youtube.com/watch?v=M59JElEPgIg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=10

## 5.11 https://www.youtube.com/watch?v=6ArSys5qHAU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=11

- When we use SoftMax function to determine the final output, for backpropagation, we use Cross Entropy to determine how well the Neural Network fits the data
- The Cross Entropy of one single row input data is the -log(max predicted probability)
- To get the total error for the Neural Network, all we do is add up the Cross Entropy values, and we can use Backpropagation to adjust the Weights and Biases and hopefully minimize the total error
- So, when the Neural Network makes a really bad prediction, Cross Entropy will help us take a relatively large step towards a better prediction
![image](https://user-images.githubusercontent.com/60442877/149275761-ea75c9a9-f1ee-4273-b570-9826348222a6.png)


