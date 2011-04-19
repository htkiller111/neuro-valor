package pl.edu.agh.miss.neuroValor.nodes;

import java.util.List;

import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;

public class InnerNeuron extends Neuron {

	private double outputCache;
	
	public InnerNeuron(DifferentiableFunction activation, List<Synapse> inputs) {
		super(activation, inputs);
	}
	
	public void cacheCurrentOutput() {
		outputCache = computeOutput();
	}

	@Override
	public double getOutput() {
		return outputCache;
	}
}