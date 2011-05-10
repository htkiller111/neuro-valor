package pl.edu.agh.miss.neuroValor.helpers;

import java.util.Arrays;

public class LearningCase {

	private final double[] inputs;
	private final double[] outputs;

	public LearningCase(double[] inputs, double[] outputs) {
		this.inputs = inputs;
		this.outputs = outputs;
	}

	public double[] getInputs() {
		return inputs;
	}

	public double[] getOutputs() {
		return outputs;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(inputs)+" -> "+Arrays.toString(outputs); 
	}

}
