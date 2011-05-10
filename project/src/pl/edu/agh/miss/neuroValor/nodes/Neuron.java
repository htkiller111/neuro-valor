package pl.edu.agh.miss.neuroValor.nodes;

import java.util.List;

import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;

public class Neuron implements OutputProducer {

	protected final List<Synapse> inputs;
	protected final DifferentiableFunction activation;
	protected double threshold;
	private final double momentum;
	private final double learningSpeed;
	
	public Neuron(DifferentiableFunction activation, List<Synapse> inputs, double momentum, double learningSpeed) {
		this.activation = activation;
		this.inputs = inputs;
		this.threshold = 0.25+Math.random()*0.75;
		this.momentum = momentum;
		this.learningSpeed = learningSpeed;
	}
	
	protected double gatherInput() {
		double ret = 0.0f;
		for (Synapse i: inputs) {
			ret += i.getWeight()*i.getFrom().getOutput();
		}
		return ret;
	}

	public double getThreshold() {
		return threshold;
	}
	
	public List<Synapse> getSynapses() {
		return inputs;
	}
	
	public DifferentiableFunction getActivation() {
		return activation;
	}
	
	@Override
	public double getOutput() {
		return computeOutput();
	}

	protected double computeOutput() {
		return activation.compute(gatherInput()+threshold);
	}

	public void setThreshold(double d) {
		this.threshold = d;
	}

	public double getMomentum() {
		return momentum;
	}

	public double getLearningRate() {
		return learningSpeed;
	}
}
