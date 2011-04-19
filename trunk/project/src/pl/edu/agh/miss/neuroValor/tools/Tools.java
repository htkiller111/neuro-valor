package pl.edu.agh.miss.neuroValor.tools;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Panel;
import java.awt.ScrollPane;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Tools {

	public static class IntArray {

		public int[] value;
		
		public IntArray(int[] value) {
			this.value = value;
		}

		@Override
		public String toString() {
			return Arrays.toString(value);
		}
		
		@Override
		public boolean equals(Object other) {
			return Arrays.equals(value, ((IntArray) other).value);
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(value);
		}
		
	}
	
	public static class FloatArray {

		public float[] value;
		
		public FloatArray(float[] value) {
			this.value = value;
		}

		@Override
		public boolean equals(Object other) {
			return Arrays.equals(value, ((FloatArray) other).value);
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(value);
		}
		
		@Override
		public String toString() {
			return Arrays.toString(value);
		}
	}

	private static final float[] CHANGE_LEVELS = {0.1f/100}; //{0.1f/100, 0.5f/100, 1.0f/100, 2.0f/100, 5.0f/100, 10.0f/100};
	
	private static final int PROBABILITY_INDICATOR_LENGTH = 50;
	private static final int LEGEND_WIDTH = 80;
	private static final int LEGEND_FONT_HEIGHT = 20;
		
	public static float[] loadMSTToSubsequentValues(File file) throws IOException {
		List<Float> vs = new ArrayList<Float>();
		BufferedReader r = new BufferedReader(new FileReader(file));
		while (true) {
			String s = r.readLine();
			if (s == null) {
				break;
			}
			if (!s.startsWith("<")) {
				vs.add(Float.valueOf(s.split(",")[5]));
			}
		}
		float[] ret = new float[vs.size()];
		for (int i=0; i<vs.size(); ++i) {
			ret[i] = vs.get(i);
		}
		return ret;
	}

	public static void showPlot(final BufferedImage image, String caption) {
		final Frame w = new Frame(caption);
		w.setLayout(new BorderLayout());
		Panel p = new Panel() {

			private static final long serialVersionUID = -8832335807893631026L;
			
			@Override
			public void paint(Graphics g) {
				g.drawImage(image, 0, 0, null);
			}
		};
		p.setPreferredSize(new Dimension(image.getWidth(), image.getHeight()));
		ScrollPane sp = new ScrollPane();
		sp.add(p);
		sp.setPreferredSize(new Dimension(image.getWidth(), image.getHeight()+20));
		w.add(sp, BorderLayout.CENTER);
		w.pack();
		w.setVisible(true);
		w.addWindowListener(new WindowListener() {

			@Override
			public void windowActivated(WindowEvent e) {				
			}

			@Override
			public void windowClosed(WindowEvent e) {				
			}

			@Override
			public void windowClosing(WindowEvent e) {
				w.dispose();
			}

			@Override
			public void windowDeactivated(WindowEvent e) {				
			}

			@Override
			public void windowDeiconified(WindowEvent e) {				
			}

			@Override
			public void windowIconified(WindowEvent e) {				
			}

			@Override
			public void windowOpened(WindowEvent e) {				
			}
		});
	}
	
	public static BufferedImage constructPlot(float[] vs, int verticalResolution) {
		BufferedImage ret = new BufferedImage(vs.length+2*LEGEND_WIDTH+LEGEND_FONT_HEIGHT, verticalResolution+2*LEGEND_FONT_HEIGHT, BufferedImage.TYPE_INT_ARGB);
		Graphics g = ret.getGraphics();
		
		float min = 1000000000.0f;
		float max = -1000000000.0f;
		for (int i=0; i<vs.length; ++i) {
			min = Math.min(vs[i], min);
			max = Math.max(vs[i], max);
		}
		
		int helpers = verticalResolution/LEGEND_FONT_HEIGHT+1;
		for (int i=0; i<helpers; ++i) {
			g.setColor(Color.BLACK);
			int placeY = verticalResolution-LEGEND_FONT_HEIGHT*(i-1);
			g.drawString(String.valueOf(min+i*(max-min)/(helpers-1)), 5, placeY+LEGEND_FONT_HEIGHT/4);
			g.drawString(String.valueOf(min+i*(max-min)/(helpers-1)), vs.length+LEGEND_WIDTH+10, placeY+LEGEND_FONT_HEIGHT/4);
			g.drawLine(LEGEND_WIDTH, placeY, vs.length+LEGEND_WIDTH-1, placeY);
		}
		int bottom = 2*verticalResolution-LEGEND_FONT_HEIGHT*(helpers-2);
		int prevY = -1;
		for (int i=0; i<vs.length; ++i) {
			int y = (int) (0.5f+bottom-verticalResolution*(vs[i]-min)/(max-min));
			if (prevY != -1) {
				g.setColor(new Color(0.2f, 0.2f, 0.9f, 0.8f));
				g.drawLine(LEGEND_WIDTH+i-1, prevY, LEGEND_WIDTH+i, y);
			}
			prevY = y;
			g.setColor(new Color(0.2f, 0.2f, 0.8f, 0.5f));
			g.drawLine(LEGEND_WIDTH+i, bottom, LEGEND_WIDTH+i, y);
		}
		g.dispose();
		
		return ret;
	}
	
	public static float prognoseJKSSA(float[] vs) {
		
		List<Map<IntArray, int[]>> afters = new ArrayList<Map<IntArray, int[]>>();
		afters.add(null);
		int[] vsd = computeChangeLevels(vs);
		int maxSubstring = 1;
		while (true) {
			Map<IntArray, int[]> after = new HashMap<IntArray, int[]>();
			afters.add(after);
			IntArray lasts = new IntArray(Arrays.copyOfRange(vsd, vsd.length-maxSubstring, vsd.length));
			boolean found = false;
			for (int d=0; d<vsd.length-maxSubstring-1; ++d) {
				IntArray nia = new IntArray(Arrays.copyOfRange(vsd, d, d+maxSubstring));
				if (!found && lasts.equals(nia)) {
					found = true;
				}
				int[] current = after.get(nia);
				if (current == null) {
					current = new int[2*CHANGE_LEVELS.length+1];
					after.put(nia, current);
				}
				++current[vsd[d+maxSubstring+1]+CHANGE_LEVELS.length];
			}
			if (!found) {
				break;
			}
			++maxSubstring;
		}

		if (maxSubstring == 1) {
			return 0.0f;
		}
		
		int[] totalFreqs = new int[2*CHANGE_LEVELS.length+1];
		for (int i=1; i<maxSubstring; ++i) {
			IntArray lasts = new IntArray(Arrays.copyOfRange(vsd, vsd.length-i, vsd.length));
			int[] freqs = afters.get(i).get(lasts);
			if (freqs != null) {
				for (int j=0; j<freqs.length; ++j) {
					totalFreqs[j] += freqs[j]*i;
				}
			}
		}
		
		float[] totalProbs = computeProbabilities(totalFreqs);
		
		float expected = 0.0f;
		for (int i=0; i<totalProbs.length; ++i) {
			expected += getMeanForChangeLevel(i-CHANGE_LEVELS.length)*totalProbs[i];
		}
		
		return expected;
	}
	
	private static float[] computeProbabilities(int[] frequencies) {
		float[] ret = new float[frequencies.length];
		float sum = 0.0f;
		for (int i=0; i<frequencies.length; ++i) {
			sum += frequencies[i];
		}
		for (int i=0; i<frequencies.length; ++i) {
			ret[i] = frequencies[i]/sum; 
		}
		return ret;
	}

	public static float[] derive(float[] vs) {
		float[] ret = new float[vs.length-1];
		for (int i=0; i<ret.length; ++i) {
			ret[i] = vs[i+1]-vs[i];
		}
		return ret;
	}
	
	public static int[] computeChangeLevels(float[] vs) {
		int[] ret = new int[vs.length-1];
		for (int i=1; i<ret.length; ++i) {
			ret[i] = whichChangeLevel(vs[i]/vs[i-1]-1);
		}
		return ret;
	}

	public static float prognoseJKCPA(float[] vs) {
		
		int shiftedChanges[] = new int[vs.length-1];
		int[] changesProbabilities = new int[2*CHANGE_LEVELS.length+1];
		
		int[][] changesAfter1Probabilities = new int[2*CHANGE_LEVELS.length+1][2*CHANGE_LEVELS.length+1];
		
		float previous = vs[0];
		for (int i=1; i<vs.length; ++i) {
			float next = vs[i];
			int change = whichChangeLevel((next-previous)/previous);
			int shiftedChange = change+CHANGE_LEVELS.length;
			++changesProbabilities[shiftedChange];
			shiftedChanges[i-1] = shiftedChange;
			previous = next;
		}
		
		int previousChange = shiftedChanges[0];
		for (int i=1; i<shiftedChanges.length; ++i) {
			int nextChange = shiftedChanges[i];
			++changesAfter1Probabilities[previousChange][nextChange];
			previousChange = nextChange;
		}
		
		for (int i=0; i<2*CHANGE_LEVELS.length+1; ++i) {
			
			int sum = 0;
			for (int j=0; j<2*CHANGE_LEVELS.length+1; ++j) {
				sum += changesAfter1Probabilities[j][i];
			}
		}
		
		int lastChange = shiftedChanges[shiftedChanges.length-1];
		
		float ret = 0.0f;
		
		for (int i=0; i<2*CHANGE_LEVELS.length+1; ++i) {
			int sum = 0;
			for (int j=0; j<2*CHANGE_LEVELS.length+1; ++j) {
				sum += changesAfter1Probabilities[i][j];
			}
			System.out.println(printProbability(changesAfter1Probabilities[i][lastChange]/(float) sum)+" chance of "+(i-CHANGE_LEVELS.length));
			ret = (i-CHANGE_LEVELS.length)*(changesAfter1Probabilities[i][lastChange]/(float) sum);
		}
		float rc = whichRealChange(ret);
		return vs[vs.length-1]*(1.0f+rc);
	}

	private static float whichRealChange(float changeMix) {
		int f = (int) Math.floor(changeMix);
		int c = (int) Math.ceil(changeMix);
		float ff = Math.abs(f-changeMix);
		float cf = Math.abs(c-changeMix);
		return getMeanForChangeLevel(f)*ff+getMeanForChangeLevel(c)*cf;
	}

	private static float getMeanForChangeLevel(int c) {
		if (c == 0) {
			return 0.0f;
		} else {
			if (c < 0) {
				c = -c;
				if (c == CHANGE_LEVELS.length) {
					return -CHANGE_LEVELS[CHANGE_LEVELS.length-1]*1.5f;
				} else {
					return -(CHANGE_LEVELS[c-1]+CHANGE_LEVELS[c])/2.0f;
				}
			} else {
				if (c == CHANGE_LEVELS.length) {
					return CHANGE_LEVELS[CHANGE_LEVELS.length-1]*1.5f;
				} else {
					return (CHANGE_LEVELS[c-1]+CHANGE_LEVELS[c])/2.0f;
				}
			}
		}
	}

	private static String printProbability(float d) {
		String ret = "";
		for (int i=0; i<d*PROBABILITY_INDICATOR_LENGTH; ++i) {
			ret += "|";
		}
		while (ret.length() < PROBABILITY_INDICATOR_LENGTH) {
			ret += " ";
		}
		return "["+ret+"]";
	}

	private static int whichChangeLevel(float d) {
		if (d < 0.0) {
			for (int i=0; i<CHANGE_LEVELS.length; ++i) {
				if (d > -CHANGE_LEVELS[i]) {
					return -i;
				}
			}
			return -CHANGE_LEVELS.length;
		} else {
			for (int i=0; i<CHANGE_LEVELS.length; ++i) {
				if (d < CHANGE_LEVELS[i]) {
					return i;
				}
			}
			return CHANGE_LEVELS.length;
		}
	}

	public static float[] toFloatArray(int[] a) {
		float[] ret = new float[a.length];
		for (int i=0; i<a.length; ++i) {
			ret[i] = a[i];
		}
		return ret;
	}

	public static String formatChange(float c) {
		float ret = ((int) 10000*c)/100.0f;
		if (ret > 0) {
			return "+"+ret+"%";
		} else {
			return ret+"%";
		}
	}

	public static String formatChange(float f, float g) {
		return formatChange(f/g-1);
	}

}