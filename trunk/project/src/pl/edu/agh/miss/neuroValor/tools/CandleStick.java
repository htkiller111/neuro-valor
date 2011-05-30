package pl.edu.agh.miss.neuroValor.tools;

import java.io.Serializable;

public class CandleStick implements Serializable {

	private static final long serialVersionUID = -8787726179364163626L;

	private static final double CHANGE_THRESHOLD = 0.01;
	public static final int FACTORS = 6;
	
	private final double open;
	private final double high;
	private final double low;
	private final double close;
	private final double vol;
	private final CandleStick former;

	public CandleStick(double open, double high, double low, double close, double vol, CandleStick former) {
		this.open = open;
		this.high = high;
		this.low = low;
		this.close = close;
		this.vol = vol;
		this.former = former;
	}

	public double getLowerSameHigher() {
		double change = close/former.getCloseValue()-1.0;
		if (change > CHANGE_THRESHOLD) {
			return 1.0;
		} else {
			if (change < -CHANGE_THRESHOLD) {
				return 0.0;
			} else {
				return 0.5;
			}
		}
	}
	
	public double getBlackDojiWhite() {
		return open > close ? 0.0 : (open < close ? 1.0 : 0.5);
	}
	
	public double getBodyShortNeutralLong() {
		return 3*getBodyLength() <= former.getBodyLength() ? 0.0 : (getBodyLength() >= 3*former.getBodyLength() ? 1.0 : 0.5);
	}

	public double getLowShadeShortNeutralLong() {
		return 3*getLowShade() <= getBodyLength() ? 0.0 : (getLowShade() >= 3*getBodyLength() ? 1.0 : 0.5);
	}
	
	public double getHighShadeShortNeutralLong() {
		return 3*getHighShade() <= getBodyLength() ? 0.0 : (getHighShade() >= 3*getBodyLength() ? 1.0 : 0.5);
	}

	public double getVolumeLowNeutralHigh() {
		return 2*vol <= former.vol ? 0.0 : (vol >= 2*former.vol ? 1.0 : 0.5);
	}
	
	private double getHighShade() {
		return high - getHigher();
	}

	private double getLowShade() {
		return getLower() - low;
	}

	private double getLower() {
		return Math.min(open, close);
	}

	private double getHigher() {
		return Math.max(open, close);
	}
	
	private double getBodyLength() {
		return Math.abs(open-close);
	}

	public double getCloseValue() {
		return close;
	}

	public double getOpenValue() {
		return open;
	}
	
	public double getHighValue() {
		return high;
	}
	
	public double getLowValue() {
		return low;
	}
	
	public double getVolumeValue() {
		return vol;
	}

	public CandleStick getCloned() {
		return new CandleStick(open, high, low, close, vol, null);
	}

}
