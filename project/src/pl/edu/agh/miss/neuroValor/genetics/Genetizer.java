package pl.edu.agh.miss.neuroValor.genetics;

import java.io.IOException;
import java.io.Serializable;
import java.net.UnknownHostException;
import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.jkgh.dee.client.ConcurrentTask;
import com.jkgh.dee.client.RemoteComputationDispatcherConnection;

public class Genetizer<T extends Evolvable<T>> {

	public static class EvolvableFitness<L> implements Comparable<EvolvableFitness<L>>, Serializable {

		private static final long serialVersionUID = -9050975364160928566L;

		private final L evolvable;
		private final double fitness;

		public EvolvableFitness(L e, double fitness) {
			this.evolvable = e;
			this.fitness = fitness;
		}

		@Override
		public int compareTo(EvolvableFitness<L> o) {
			return Double.compare(o.getFitness(), fitness);
		}

		public L getEvolvable() {
			return evolvable;
		}

		public double getFitness() {
			return fitness;
		}
		
		@Override
		public String toString() {
			return evolvable+" ("+fitness+")";
		}
	}

	public static class FitnessEvaluationTask<T extends Evolvable<T>> implements ConcurrentTask<EvolvableFitness<T>> {

		private static final long serialVersionUID = 453453451L;
		
		private final T child;

		public FitnessEvaluationTask(T child) {
			this.child = child;
		}

		@Override
		public EvolvableFitness<T> execute() {
			return new EvolvableFitness<T>(child, child.computeFitness());
		}
		
	}
	
	private final GeneticAlgorithmSettings settings;
	private List<EvolvableFitness<T>> fitnesses;
	private final RemoteComputationDispatcherConnection computationServer;

	public Genetizer(GeneticAlgorithmSettings settings, List<T> generation, RemoteComputationDispatcherConnection dispatcherConnection) throws UnknownHostException, IOException, ClassNotFoundException {
		this.settings = settings;
		this.computationServer = dispatcherConnection;

		List<ConcurrentTask<EvolvableFitness<T>>> tasks = new ArrayList<ConcurrentTask<EvolvableFitness<T>>>();
		for (T e: generation) {
			tasks.add(new FitnessEvaluationTask<T>(e));
		}
		
		fitnesses = computationServer.execute(tasks);
	}
	
	public void step() throws RemoteException {
		
		Collections.sort(fitnesses);
		int selected = (int) (settings.getSelectionRatio()*fitnesses.size());
		final List<EvolvableFitness<T>> nextGeneration = new ArrayList<EvolvableFitness<T>>();
		List<EvolvableFitness<T>> parents = fitnesses.subList(0, selected);
		nextGeneration.addAll(parents);
		
		List<ConcurrentTask<EvolvableFitness<T>>> tasks = new ArrayList<ConcurrentTask<EvolvableFitness<T>>>();
		for (int i=selected; i<fitnesses.size(); ++i) {
			T child = pickRandomFrom(fitnesses.subList(0, selected)).getEvolvable().copied();
			if (Math.random() < settings.getCrossoverChance()) {
				T secondParent = pickRandomFrom(parents).getEvolvable().copied();
				for (int index: pickRandomIndices(child.getGenes().size(), child.getGenes().size()/2)) {
					child.getGenes().set(index, secondParent.getGenes().get(index));
				}
			}
			for (Gene g: child.getGenes()) {
				if (Math.random() < settings.getMutationChance()) {
					g.mutateBy(2*(0.5-Math.random())*settings.getMaxMutationChange());
				}
			}
			tasks.add(new FitnessEvaluationTask<T>(child));
		}
		nextGeneration.addAll(computationServer.execute(tasks));
		fitnesses = nextGeneration;
	}

	public List<EvolvableFitness<T>> getGenerationFitnesses() {
		return fitnesses;
	}
	
	public static <E> E pickRandomFrom(List<E> parents) {
		return parents.get((int) (Math.random()*parents.size()));
	}

	public static int[] pickRandomIndices(int all, int picked) {
		int[] ret = new int[picked];
		boolean[] helpers = new boolean[all];
		for (int i=0; i<picked; ++i) {
			int which = (int) (Math.random()*(all-i)) + 1;
			for (int j=0; j<all; ++j) {
				if (!helpers[j]) {
					--which;
					if (which == 0) {
						helpers[j] = true;
						ret[i] = j;
						break;
					}
				}
			}
		}
		return ret;
	}
}
