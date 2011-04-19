package pl.edu.agh.miss.neuroValor.main;

import java.util.ArrayList;
import java.util.List;

import pl.edu.agh.miss.neuroValor.NeuronNet;

public class Main {

	public static void main(String[] args) {
		
		NeuronNet nn = new NeuronNet(2, 1, 3);
		
		List<double[]> learningSet = new ArrayList<double[]>();
		for (int i=0; i<2000000; ++i) {
			double arg0 = Math.random();
			double arg1 = Math.random();
			double ret0 = (arg0 + arg1) / 2;
			learningSet.add(new double[] {ret0, arg0, arg1});
		}
		
		double sumError = 0.0;
		for (int i=0; i<learningSet.size(); ++i) {
			double[] lc = learningSet.get(i);
			nn.learnOn(new double[] {lc[0]}, lc[1], lc[2]);
			sumError += Math.abs(nn.compute(lc[1], lc[2])[0]-lc[0]);
			if (sumError/i < 0.01) {
				System.out.println("Converged.");
				break;
			}
			if (i%(learningSet.size()/10) == 0) {
				System.out.println(i);
			}
		}
		
		System.err.println(nn.compute(0.5, 0.5)[0]);
		System.err.println(nn.compute(0.7, 0.2)[0]);
		System.err.println(nn.compute(0.2, 0.7)[0]);
		System.err.println(nn.compute(0.99, 0.99)[0]);
		System.err.println(nn.compute(0, 0)[0]);
		System.err.println(nn.compute(0.3, 0.2)[0]);
	}
}
