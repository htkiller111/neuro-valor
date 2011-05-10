package pl.edu.agh.miss.neuroValor.genetics;

import java.io.Serializable;

public interface Gene extends Serializable {

	public void mutateBy(double ratio);

}
