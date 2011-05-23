package pl.edu.agh.miss.neuroValor.genetics;

import java.util.Arrays;
import java.util.List;

import pl.edu.agh.miss.neuroValor.tools.Tools;

public class BasicNetworkStructure implements Evolvable<BasicNetworkStructure> {

	private static final long serialVersionUID = 342342342341L;

	private final BasicNetworkEvaluator e;

	private final SingleNaturalNumberGene inputCountGene;
	private final SingleNaturalNumberGene firstCountGene;
	private final SingleNaturalNumberGene secondCountGene;
	private final SingleBooleanGene usingTanhGene;

	public BasicNetworkStructure(BasicNetworkEvaluator e) {
		this.e = e;
		this.inputCountGene = new SingleNaturalNumberGene(Tools.random(1, 200), 200);
		this.firstCountGene = new SingleNaturalNumberGene(Tools.random(1, 400), 400);
		this.secondCountGene = new SingleNaturalNumberGene(Tools.random(1, 400), 400);
		this.usingTanhGene = new SingleBooleanGene(Math.random() < 0.5);
	}
	
	public BasicNetworkStructure(BasicNetworkEvaluator e, int inputCount, int firstCount, int secondCount, boolean usingTanh) {
		this.e = e;
		this.inputCountGene = new SingleNaturalNumberGene(inputCount, 200);
		this.firstCountGene = new SingleNaturalNumberGene(firstCount, 400);
		this.secondCountGene = new SingleNaturalNumberGene(secondCount, 400);
		this.usingTanhGene = new SingleBooleanGene(usingTanh);
	}
	
	@Override
	public double computeFitness() {
		return e.evalute(this);
	}

	@Override
	public BasicNetworkStructure copied() {
		return new BasicNetworkStructure(e, getInputCount(), getFirstCount(), getSecondCount(), isUsingTanh());
	}

	@Override
	public List<Gene> getGenes() {
		return Arrays.asList(inputCountGene, firstCountGene, secondCountGene, usingTanhGene);
	}

	public int getInputCount() {
		return inputCountGene.getValue();
	}

	public int getFirstCount() {
		return firstCountGene.getValue();
	}
	
	public int getSecondCount() {
		return secondCountGene.getValue();
	}
	
	public boolean isUsingTanh() {
		return usingTanhGene.getValue();
	}

	@Override
	public void copyGene(BasicNetworkStructure from, int index) {
		switch (index) {
		case 0:
			inputCountGene.setValue(from.getInputCount());
			break;
		case 1:
			firstCountGene.setValue(from.getFirstCount());
			break;
		case 2:
			secondCountGene.setValue(from.getSecondCount());
			break;
		case 3:
			usingTanhGene.setValue(from.isUsingTanh());
			break;
		}
	}
	
	@Override
	public String toString() {
		return inputCountGene.getValue()+" -> "+firstCountGene.getValue()+" -> "+secondCountGene.getValue()+" -> 1 ("+(usingTanhGene.getValue() ? "Tanh" : "Sigmoid")+")";
	}
}
