package pl.edu.agh.miss.neuroValor.nodes;

import java.util.List;

import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;

public class Neuron implements OutputProducer {

	protected List<Synapse> inputs;
	protected DifferentiableFunction activation;
	protected double threshold;
	
	public Neuron(DifferentiableFunction activation, List<Synapse> inputs) {
		this.activation = activation;
		this.inputs = inputs;
		this.threshold = Math.random()*0.75+0.25;
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
}
