package pl.edu.agh.miss.neuroValor.genetics;

public class SingleNaturalNumberGene implements Gene {

	private static final long serialVersionUID = 8025333209116624393L;

	private int value;
	private int range;

	public SingleNaturalNumberGene(int value, int range) {
		this.value = value;
		this.range = range;
	}
	
	@Override
	public void mutateBy(double ratio) {
		value += (int) (ratio*range);
		if (value < 1) {
			value = 1;
		}
	}

	public int getValue() {
		return value;
	}

}
