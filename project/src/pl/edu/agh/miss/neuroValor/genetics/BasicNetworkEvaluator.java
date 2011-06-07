package pl.edu.agh.miss.neuroValor.genetics;

import java.io.Serializable;

public interface BasicNetworkEvaluator extends Serializable {

	public BasicNetworkStats getBasicNetworkStats();

	public double evalute(BasicNetworkStructure basicNetworkStructure);

}
