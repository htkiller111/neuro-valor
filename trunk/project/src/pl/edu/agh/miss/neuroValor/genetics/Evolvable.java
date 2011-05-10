package pl.edu.agh.miss.neuroValor.genetics;

import java.io.Serializable;
import java.util.List;

public interface Evolvable<T extends Evolvable<T>> extends Serializable {

	public double computeFitness();
	public List<Gene> getGenes();
	public T copied();

}
