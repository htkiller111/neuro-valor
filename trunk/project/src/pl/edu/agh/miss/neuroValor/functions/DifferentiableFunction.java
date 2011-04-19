package pl.edu.agh.miss.neuroValor.functions;

public interface DifferentiableFunction extends Function {

	public static DifferentiableFunction SIGMOID_ONE = new DifferentiableFunction() {

		@Override
		public double compute(double x) {
			return 1.0/(1.0+Math.exp(-x));
		}

		@Override
		public double deriveComputation(double cx) {
			return cx*(1-cx);
		}
		
	};

	/**
	 * Computes own derivative based on own value. 
	 * @param cx a value of this function for some input x
	 * @return a value of derivative of this function for input x
	 */
	public double deriveComputation(double cx);
}
