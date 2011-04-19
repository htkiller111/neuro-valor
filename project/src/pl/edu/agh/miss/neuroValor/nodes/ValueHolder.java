package pl.edu.agh.miss.neuroValor.nodes;

public class ValueHolder implements OutputProducer {

	private double output;

	@Override
	public double getOutput() {
		return output;
	}

	public void setValue(double output) {
		this.output = output;
	}

}
