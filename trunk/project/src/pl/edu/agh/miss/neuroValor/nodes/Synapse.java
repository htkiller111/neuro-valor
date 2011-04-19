package pl.edu.agh.miss.neuroValor.nodes;

public class Synapse {

	private double weight;
	private OutputProducer from;
	
	public Synapse(OutputProducer from) {
		this.weight = Math.random()-0.5;
		this.from = from;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}

	public double getWeight() {
		return weight;
	}

	public OutputProducer getFrom() {
		return from;
	}
	
}
