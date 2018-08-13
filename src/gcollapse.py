#!/usr/bin/env python3

from boundary import *

def flxwhom(xs, flx, eg):
    """Flux weighted homogenization.

    xs = A list of cs values per eg
    flx = A list of flux weights
    eg = Group collapse lists

    """

    return [sum([xs[g]*flx[g] for g in MG])/sum([flx[g] for g in MG]) for MG in eg]

def flxwscathom(sxs, flx, eg):
    """Flux weighted scattering xs homogenization.

    xs = A list of lists. Each list is the scattering xs from group g into group g' 
         if g is the first index and g' is the second.
    flx = A list of flux weights
    eg = Group collapse lists

    """

    hxs = [[0.0 for g in range(len(eg))] for g in range(len(eg))]
    for ig, gli in enumerate(eg):
        for og, glo in enumerate(eg):
            hxs[ig][og] = (sum([sum([sxs[g,x] for x in glo])*flx[g] for g in gli]) /
                           sum([flx[g] for g in gli]))
    return hxs

def gsum(v,eg):
    return [sum([v[g] for g in gl]) for gl in eg]

def write_data(fname, T, S, F, D):
    with open(fname, 'w') as of:
        for g in range(len(T)):
            of.write('%.12e %.12e 0.0 %.12e ' % (F[g], T[g], D[g]) +
                     ' '.join(['%.12e' % S[g][go] for go in range(len(S[g]))]) + '\n')

if __name__ == '__main__':
    from argparse import ArgumentParser as AP

    parser = AP(description="Solves the very specific problem")
    parser.add_argument('-i', nargs=2, default=['input/leftreg.dat.csv', 'input/rightreg.dat.csv'],
                        help="CMM tally results")
    parser.add_argument('-e', default='input/EnergyGroup.dat', help="Energy group structure input")
    parser.add_argument('-eo', default='input/EnergyGroup_2g.dat',
                        help="Energy group structure homogenized")
    parser.add_argument('-l', default=0.738, type=float,
                        help="Percentage of neutrons that leaked in OpenMC")
    parser.add_argument('-o', nargs=2, default=['input/leftreg_2g.csv', 'input/rightreg_2g.csv'],
                        help="Output files")
    parser.add_argument('-g', nargs='+', default=[25,70], help="Collapsed energy group boundaries")

    args = parser.parse_args()

    regions = []
    Eg = ([[g for g in range(0, args.g[0])]] +
          [[g for g in range(g1, g2)] for g1,g2 in zip(args.g[:-1], args.g[1:])])
    for fi, fo in zip(args.i, args.o):
        inp = read_inp(fi)
        sigT, sigS, flx, D = inp
        sTC = flxwhom(sigT, flx, Eg)
        sSC = flxwscathom(sigS, flx, Eg)
        rJ = get_tallied(flx, D)
        rJC = gsum(rJ,Eg)
        flxC = gsum(flx,Eg)
        DC = [rJC[g] / (3.0 * flxC[g]) for g in range(len(rJC))]
        write_data(fo, sTC, sSC, flxC, DC)
    with open(args.e, 'r') as f:
        lines = f.readlines()[1:] #Skip header
    eb = np.array([float(v.strip()) for v in lines])
    ebC = [eb[0]] + [eb[g] for g in args.g]
    with open(args.eo, 'w') as f:
        f.write('#Group structure, boundaries in eV\n')
        for v in ebC:
            f.write('%.12e\n' % v)

    
