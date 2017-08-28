import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class FileHandle {

	int NODES = 784;
	BufferedReader reader;
	String file_name;
	String sample_data;
	double[] sample;
	int count;

	public FileHandle(String file_name) {
		this.file_name = file_name;
		sample_data = "";
		sample = new double[NODES + 1];
		count = 0;
	}

	// open the file
	public void open_file() throws FileNotFoundException {
		reader = new BufferedReader(new FileReader(file_name));
	}

	// close the file
	public void close_file() throws IOException {
		reader.close();
	}

	// read the next sample and return true if another sample is a vailable
	// return false if no more data (null)
	public boolean sample_available() throws IOException {
		sample_data = reader.readLine();
		if (sample_data != null) {
			return true;
		} else {
			return false;
		}
	}

	// prepare the sample array with data from file
	public double[] get_sample() {
		String[] temp_nodes = sample_data.split(",");

		// this is the value of the sample
		sample[0] = Double.parseDouble(temp_nodes[0]);

		// the rest is the pixel values of the sample / 255
		for (int i = 1; i < temp_nodes.length; i++) {
			sample[i] = (Double.parseDouble(temp_nodes[i])) / 255;
		}

		return sample;
	}
}
