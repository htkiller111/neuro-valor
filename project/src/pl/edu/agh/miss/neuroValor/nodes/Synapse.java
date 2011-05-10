package pl.edu.agh.miss.neuroValor.nodes;

public class Synapse {

	private double weight;
	private OutputProducer from;
	private double previousChange;
	
	public Synapse(OutputProducer from) {
		this.weight = Math.random()-0.5;
		this.from = from;
		this.previousChange = 0;
	}

	public OutputProducer getFrom() {
		return from;
	}

	public void changeWeight(double d, double momentum) {
		double thisChange = d + momentum*previousChange;
		weight += thisChange;
		previousChange = thisChange;
	}

	public double getWeight() {
		return weight;
	}
	
}
