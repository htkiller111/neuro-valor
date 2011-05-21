package pl.edu.agh.miss.neuroValor.genetics;

public class SinglePositiveNumberGene implements Gene {

	private static final long serialVersionUID = 2403575590360647155L;

	private double v;

	public SinglePositiveNumberGene(double v) {
		this.v = v;
	}

	@Override
	public void mutateBy(double ratio) {
		v += v*ratio;
		if (v<0.0) {
			v = 0.0;
		}
	}

	public double getValue() {
		return v;
	}
	
	@Override
	public String toString() {
		return Double.toString(v);
	}

	public void setValue(double value) {
		v = value;
	}

}
