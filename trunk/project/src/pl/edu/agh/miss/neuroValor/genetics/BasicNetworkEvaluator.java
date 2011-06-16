package pl.edu.agh.miss.neuroValor.genetics;

import java.io.Serializable;

public interface BasicNetworkEvaluator extends Serializable {

	public double evalute(BasicNetworkStructure basicNetworkStructure);

}
