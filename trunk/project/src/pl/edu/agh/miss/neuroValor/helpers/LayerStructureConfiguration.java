package pl.edu.agh.miss.neuroValor.helpers;

import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;


public class LayerStructureConfiguration {

	private double momentum;
	private double learningRate;
	private int neuronCount;
	private DifferentiableFunction activation;

	public LayerStructureConfiguration() {	
	}
	
	public LayerStructureConfiguration(double momentum, double learningRate, int neuronCount, DifferentiableFunction activation) {
		this.momentum = momentum;
		this.learningRate = learningRate;
		this.neuronCount = neuronCount;
		this.activation = activation;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public int getNeuronCount() {
		return neuronCount;
	}

	public void setNeuronCount(int neuronCount) {
		this.neuronCount = neuronCount;
	}

	public DifferentiableFunction getActivation() {
		return activation;
	}

	public void setActivation(DifferentiableFunction activation) {
		this.activation = activation;
	}
	
}
