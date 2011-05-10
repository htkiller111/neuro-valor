package pl.edu.agh.miss.neuroValor.genetics;

public class GeneticAlgorithmSettings {

	private final double selectionRatio;
	private final double crossoverChance;
	private final double mutationChance;
	private final double maxMutationChange;
	
	public GeneticAlgorithmSettings(double selectionRatio,
			double crossoverChance, double mutationChance,
			double maxMutationChange) {
		this.selectionRatio = selectionRatio;
		this.crossoverChance = crossoverChance;
		this.mutationChance = mutationChance;
		this.maxMutationChange = maxMutationChange;
	}

	public double getSelectionRatio() {
		return selectionRatio;
	}

	public double getCrossoverChance() {
		return crossoverChance;
	}

	public double getMutationChance() {
		return mutationChance;
	}

	public double getMaxMutationChange() {
		return maxMutationChange;
	}

}
