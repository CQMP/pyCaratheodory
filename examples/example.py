import sys
sys.path.insert(0, '../caratheodory')
# the above two lines are used to import the caratheodory package, you can also install the package and import it directly
from caratheodory import Caratheodory
import numpy as np
import argparse

def interpolate(G, omega, beta, beta_new, kernel, bound=50):
    # Due to the accumulated numerical errors in calculating S, interpolating with reversed omega and G could be more accurate for large frequencies.
    # The ``bound" variable is used to determine when to switch from the normal order to the reversed order.
    # In most time, (when the number of frequencies is not large), this does not matter.
    G_new = np.zeros(G.shape, dtype=np.complex128)
    omega_new = omega*beta/beta_new
    kernel.build(omega, G)
    G_new[omega_new.imag < bound, :, :] = kernel.evaluate(omega_new[omega_new.imag < bound])
    kernel.build(omega[::-1], G[::-1].copy())
    G_new[omega_new.imag >= bound, :, :] = kernel.evaluate(omega_new[omega_new.imag >= bound])
    return G_new, omega_new 


def main():
    parser = argparse.ArgumentParser(description='Interpolate matrix-valued causal functions using the Caratheodory method.')
    parser.add_argument('--inputfile', required=True, help='Input file containing  matrix-valued causal  function data')
    parser.add_argument('--beta', type=float, required=True, help='Inverse temperature for input data')
    parser.add_argument('--beta_new', type=float, required=True, help='Inverse temperature for output data')
    parser.add_argument('--outputfile', required=True, help='Output file to save the interpolated  matrix-valued causal function data')
    parser.add_argument('--bound', type=int, default=50, help='Bound for frequency reversal (default: 50)')

    args = parser.parse_args()

    data = np.loadtxt(args.inputfile, dtype=np.complex_)
    omega = data[:, 0]
    G = data[:, 1:]
    beta = args.beta
    beta_new = args.beta_new

    N = int(np.sqrt(G.shape[1]))
    if N != np.sqrt(G.shape[1]):
        print('The columns in the text file are expected to be: omega, G_{00}, G_{01}, ... G_{0(N-1)}, G_{10} ... G_{(N-1)(N-1)}, please check your input file.')
        exit()

    kernel = Caratheodory()
    G = np.reshape(G, (G.shape[0], N, N))
    G_new, omega_new = interpolate(G, omega, beta, beta_new, kernel, bound=args.bound)

    output_data = np.column_stack((omega_new, G_new.reshape(G_new.shape[0], -1)))
    np.savetxt(args.outputfile, output_data, fmt=['%.8e%+.8ej']*output_data.shape[1], delimiter='\t')

if __name__ == '__main__':
    main()
