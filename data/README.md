The directory used for storing generated data from scripts in examples.

Additionally, it contains examples of incoming wave and scatterer specifications (<tt>incoming_\*.dat</tt> and <tt>scatterer_\*.dat</tt> files, respectively).

The file <tt>parula.pal</tt> defines the Parula colormap which is used by gnuplot scripts.

To view the scatterer <tt>\<filename.dat\></tt>, enter the following command in [Octave](https://octave.org/)/Matlab terminal:
~~~
V = load('<filename.dat>'); V = [V; V(1,:)]; fill(V(:,1),V(:,2),'b'); axis square off
~~~
