set terminal epslatex size 12cm,6cm
set output "speedup.tex"

set object 1 rect from graph 0,graph 0 to graph 1,graph 1 fc rgb "#EEEEEE" fs solid 1.0 noborder behind
set grid lt 1 lw 3 lc rgb "#FAFAFA"

set xl "Subdivision $s$"
set yl "Speedup $S_\\mathrm{A} = t^\\mathrm{(CPU)}_\\mathrm{A} / t^\\mathrm{(GPU)}_\\mathrm{A}$"
plot 'speedup.txt' u 1:($2/$3) w lp lc black pt 13 title ""