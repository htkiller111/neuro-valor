package pl.edu.agh.miss.neuroValor.genetics;

public class SingleBooleanGene implements Gene {

	private static final long serialVersionUID = -1183972305702119037L;
	
	private boolean b;

	public SingleBooleanGene(boolean b) {
		this.b = b;
	}

	@Override
	public void mutateBy(double ratio) {
		b = !b;
	}

	public boolean getValue() {
		return b;
	}

	public void setValue(boolean v) {
		b = v;
	}

}
