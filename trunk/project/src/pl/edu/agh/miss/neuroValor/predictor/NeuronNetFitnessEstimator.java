package pl.edu.agh.miss.neuroValor.predictor;

import java.io.Serializable;

import pl.edu.agh.miss.neuroValor.NeuronNet;

public interface NeuronNetFitnessEstimator extends Serializable {

	public double estimate(NeuronNet nn);
}
