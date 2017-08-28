import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class Main {

	// constants
	static final int TOTAL_TRAIN = 60000;
	static final int TOTAL_TEST = 10000;
	final static int DIGITS = 10;
	final static int NODES = 784;
	// 50 training epochs
	// plus 1 epoch to get accuracy before training
	final static int EPOCHS = 51;
	final static double LEARN_RATE = .1;

	// number of hidden units (user input)
	static int hidden_units = 0;

	// momentum (user input)
	static double momentum = 0;

	// fraction of the training samples to use (user input)
	static double training_used = 0;

	// number of samples to be used
	static int training_samples = 0;

	// value to indicate if training or testing
	static boolean is_training = false;

	// Lists for testing and training samples (input to neural network)
	static List<double[]> training = new ArrayList<double[]>();
	static List<double[]> testing = new ArrayList<double[]>();

	// Lists for hidden & output nodes/weights
	static List<Node> hidden_nodes = new ArrayList<Node>();
	static List<Node> output_nodes = new ArrayList<Node>();

	// array for the hidden activations
	static double[] hidden_activations;

	// array for the output error terms
	static double[] output_errors;

	// accuracy arrays
	static int[] train_correct;
	static double[] train_accuracy;
	static int[] test_correct;
	static double[] test_accuracy;
	static int[][] test_matrix;

	public static void main(String[] args) throws IOException {
		// must be given 3 arguments to run
		// arg[0]: number of hidden units
		// arg[1]: momentum
		// arg[2]: fraction of training samples to use (decimal)
		// example, 50% of training samples = parameter of .5

		// check that user input required args
		// and that args are valid input
		if ((args.length == 3) & isValidInt(args[0]) & isValidDouble(args[1])
				& isValidDouble(args[2])) {

			// set the variables and instantiate arrays
			hidden_units = Integer.parseInt(args[0]);
			momentum = Double.parseDouble(args[1]);
			training_used = Double.parseDouble(args[2]);
			training_samples = (int) (TOTAL_TRAIN * training_used);

			hidden_activations = new double[hidden_units];
			output_errors = new double[DIGITS];
			train_correct = new int[EPOCHS];
			train_accuracy = new double[EPOCHS];
			test_correct = new int[EPOCHS];
			test_accuracy = new double[EPOCHS];
			test_matrix = new int[DIGITS][DIGITS];

			System.out.println("Running Neural Network experiment for\n"
					+ "    hidden units: " + hidden_units + "    momentum: "
					+ momentum + "    training samples: " + training_used + " "
					+ training_samples);
		} else {
			System.out
					.println("Error: provide the following three parameters"
							+ " when running this application:\n"
							+ "    1: number of hidden units to use\n"
							+ "    2: momentum during training (decimal between 0 and 1)\n"
							+ "    3: fraction of training samples to use (decimal between 0 and 1)");
			System.exit(0);
		}

		System.out.println("filling training array");

		// open the training file, get samples to add to the training array
		// list, and close the file when all samples collected
		FileHandle train_file = new FileHandle("mnist_train.csv");
		train_file.open_file();
		while (train_file.sample_available()) {
			training.add(train_file.get_sample().clone());
		}
		train_file.close_file();

		// also fill the testing sample array list
		System.out.println("filling testing array");
		FileHandle test_file = new FileHandle("mnist_test.csv");
		test_file.open_file();
		while (test_file.sample_available()) {
			testing.add(test_file.get_sample().clone());
		}
		test_file.close_file();

		// if the training samples specified is smaller then the
		// training file, get a random permutation of the samples
		// and keep only the amount requested from input
		if (training_samples < TOTAL_TRAIN) {
			shuffleList(training);
			for (int i = TOTAL_TRAIN - 1; i > training_samples - 1; i--) {
				training.remove(i);
			}
		}

		// instantiate the hidden nodes
		for (int i = 0; i < hidden_units; i++) {
			hidden_nodes.add(new Node(momentum, NODES, LEARN_RATE, true));
		}

		// instantiate the output nodes
		for (int i = 0; i < DIGITS; i++) {
			output_nodes
					.add(new Node(momentum, hidden_units, LEARN_RATE, false));
		}

		// Here the experiment begins
		// run through 50 epochs (including epoch # 0)
		for (int epoch = 0; epoch < EPOCHS; epoch++) {

			System.out.println("starting epoch: " + epoch);

			// make sure training state is set
			is_training = true;

			for (double[] sample : training) {

				// on epoch 0, do not do this step
				// since we want to collect accuracy before
				// any updates are made
				if (epoch > 0) {

					// update activation of hidden and output nodes
					activate(sample);

					// calc error terms and update weights for
					// input to hidden nodes and for
					// hidden to output nodes
					update_weights(sample);
				}

				// calculate accuracy on this sample
				find_accuracy(epoch, (int) sample[0]);

			}

			// change to testing state
			is_training = false;

			for (double[] sample : testing) {

				// update activation of hidden and output nodes
				activate(sample);

				// calc accuracy on this sample
				find_accuracy(epoch, (int) sample[0]);
			}

			// shuffle training list for new permutation on next epoch
			shuffleList(training);

		}// end epochs

		// get final accuracy calculations and fill in confusion matrix
		// print out the training and test accuracy & matrix
		finalize_accuracy();

	}

	// fuction that updates the activations for both
	// hidden and output nodes
	private static void activate(double[] sample) {
		for (int node = 0; node < hidden_units; node++) {
			hidden_nodes.get(node).activate(sample);
			hidden_activations[node] = hidden_nodes.get(node).get_activation();
		}

		for (int node = 0; node < DIGITS; node++) {
			output_nodes.get(node).activate(hidden_activations);
		}
	}

	// function that calculates error terms
	// and updates weights for the hidden and
	// output nodes
	private static void update_weights(double[] sample) {
		// calc error terms
		for (int node = 0; node < DIGITS; node++) {
			output_errors[node] = output_nodes.get(node).output_error_calc(
					sample[0] == node);
		}

		for (int node = 0; node < hidden_units; node++) {
			hidden_nodes.get(node).hidden_error_calc(output_nodes, node);
		}

		// update weights
		for (int node = 0; node < DIGITS; node++) {
			output_nodes.get(node).weight_update(hidden_activations);
			hidden_nodes.get(node).weight_update(sample);
		}
	}

	// function that computes accuracy for the epoch
	private static void find_accuracy(int epoch, int sample_val) {
		int size = 0;
		int high_idx = 0;
		double high = output_nodes.get(0).get_activation();
		size = output_nodes.size();

		// find the prediction for the sample
		for (int i = 1; i < size; i++) {
			if (output_nodes.get(i).get_activation() > high) {
				high_idx = i;
				high = output_nodes.get(i).get_activation();
			}
		}

		// update the appropriate accuracy array for given epoch
		if (high_idx == sample_val) {
			if (is_training) {
				train_correct[epoch]++;
			} else {
				test_correct[epoch]++;
			}
		}

		// update the matrix for testing on final epoch
		if ((epoch == EPOCHS - 1) & (!is_training)) {
			test_matrix[sample_val][high_idx]++;
		}
	}

	// function finalizes and prints the accuracy data
	static private void finalize_accuracy() {
		for (int i = 0; i < EPOCHS; i++) {
			train_accuracy[i] = (double) train_correct[i] / training.size();
			test_accuracy[i] = (double) test_correct[i] / testing.size();
		}

		System.out.print("Training accuracy: { ");
		for (int i = 0; i < EPOCHS; i++) {
			System.out.print(train_accuracy[i] + " ");
		}
		System.out.println("}");

		System.out.print("Testing accuracy: { ");
		for (int i = 0; i < EPOCHS; i++) {
			System.out.print(test_accuracy[i] + " ");
		}
		System.out.println("}");

		System.out.println("confusion matrix: ");
		for (int i = 0; i < DIGITS; i++) {
			for (int j = 0; j < DIGITS; j++) {
				System.out.print("    " + test_matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

	// function checks that user input is an integer
	private static boolean isValidInt(String string) {
		try {
			Integer.parseInt(string);
		} catch (NumberFormatException n) {
			return false;
		}
		return true;
	}

	// function checks that user input is between 0 and 1
	private static boolean isValidDouble(String string) {
		double x;
		try {
			x = Double.parseDouble(string);
		} catch (NumberFormatException n) {
			return false;
		}

		if (!(x < 0) & !(x > 1)) {
			return true;
		} else {
			return false;
		}
	}

	// found this shuffle list from
	// http://www.vogella.com/tutorials/JavaAlgorithmsShuffle/article.html
	// this will shuffle the training data per homework instruction
	public static void shuffleList(List<double[]> a) {
		int n = a.size();
		Random random = new Random();
		random.nextInt();
		for (int i = 0; i < n; i++) {
			int change = i + random.nextInt(n - i);
			swap(a, i, change);
		}
	}

	// The other part of the shuffle function found from
	// http://www.vogella.com/tutorials/JavaAlgorithmsShuffle/article.html
	private static void swap(List<double[]> a, int i, int change) {
		double[] helper = a.get(i);
		a.set(i, a.get(change));
		a.set(change, helper);
	}

}