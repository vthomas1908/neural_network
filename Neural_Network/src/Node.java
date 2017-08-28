import java.util.List;
import java.util.Random;

public class Node {

	// variables

	// node value
	double activation = 0;

	// momentum
	double momentum = 0;

	// learning rate
	double learn_rate = 0;

	// error value
	double error = 0;

	// weights from input (pixel or hidden) to node (hidden or output)
	double[] weights;

	// delta weights
	double[] delta_weights;

	// previous sample delta weights
	double[] prior_delta_weights;

	// size of the weight arrays
	int size = 0;

	// flag to determine if hidden or output
	boolean is_hidden;

	public Node(double momentum, int inputs, double learn_rate, boolean hidden) {
		// set the momentum
		this.momentum = momentum;

		// set the array size (+1 for bias)
		this.size = inputs + 1;

		// set the learning rate
		this.learn_rate = learn_rate;

		// set the hidden node flag
		this.is_hidden = hidden;

		// initialize arrays
		this.weights = new double[this.size];
		this.prior_delta_weights = new double[this.size];
		this.delta_weights = new double[this.size];

		// initial weights will be random between -.5 and .5
		for (int j = 0; j < this.size; j++) {
			Random rand = new Random();
			this.weights[j] = rand.nextDouble() - .5;
		}

	}

	public void activate(double[] sample) {
		// don't forget to add the bias!
		this.activation = sigmoid(dot_product(sample) + this.weights[0]);
	}

	private double dot_product(double[] sample) {
		double dp = 0;
		// calc dot product not including 0th bias value
		for (int i = 1; i < size; i++) {
			if (is_hidden) {
				// first node of input pixels is class (ignore)
				dp += this.weights[i] * sample[i];
			} else {
				// don't ignore first input hidden nodes
				dp += this.weights[i] * sample[i - 1];
			}

		}

		// note: bias added in at function call

		return dp;
	}

	// function calculates sigmoid
	private double sigmoid(double dp) {
		return 1 / (1 + Math.exp(dp * -1));
	}

	public void weight_update(double[] input_nodes) {
		// start at index 1 and add bias at end
		for (int i = 1; i < this.size; i++) {
			// get the last delta weight
			this.prior_delta_weights[i] = this.delta_weights[i];

			// this is when input nodes are the pixel units
			// ignore the first index since that is the "class"
			// calc the new delta weight
			if (is_hidden) {
				this.delta_weights[i] = this.learn_rate * this.error
						* input_nodes[i];
				this.delta_weights[i] += this.momentum
						* this.prior_delta_weights[i];
			}

			// this is when input nodes are the hidden units
			// don't forget to get index 0!
			if (!is_hidden) {
				this.delta_weights[i] = this.learn_rate * this.error
						* input_nodes[i - 1];
				this.delta_weights[i] += this.momentum
						* this.prior_delta_weights[i];
			}

			// find the new weight
			this.weights[i] += this.delta_weights[i];
		}

		// don't forget about the bias!
		this.prior_delta_weights[0] = this.delta_weights[0];
		this.delta_weights[0] = this.learn_rate * this.error;
		this.delta_weights[0] += this.momentum * this.prior_delta_weights[0];
		this.weights[0] += this.delta_weights[0];

	}

	// function to calc error term for output nodes
	public double output_error_calc(boolean is_target) {
		double target;
		
		// is this the target "class"
		if (is_target) {
			target = .9;
		} else {
			target = .1;
		}
		
		// here is the error
		error = activation * (1 - activation) * (target - activation);
		
		return error;
	}

	// function to calc error term for hidden nodes
	public void hidden_error_calc(List<Node> nodes, int index) {
		error = activation * (1 - activation) * (sum_prod(nodes, index));
	}

	// only calculated on hidden nodes
	// calc the sum of (weight from this hidden node to each output node
	// times the error at that output node)
	private double sum_prod(List<Node> nodes, int index) {
		int len = nodes.size();
		double val = 0;
		
		for (int i = 0; i < len; i++) {
			
			// don't want the bias weight in this calc (index + 1)
			val += nodes.get(i).get_weights()[index + 1]
					* nodes.get(i).get_error();
		}

		return val;
	}

	// return the activation of this node
	public double get_activation() {
		return this.activation;
	}

	// return the array of weights coming into this node
	public double[] get_weights() {
		return this.weights;
	}

	// return the error term at this node
	public double get_error() {
		return error;
	}

}
