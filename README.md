# flocra-pulseq
Pulseq interpreter for vnegnev's flow-based OCRA (FLOCRA)

# Usage:
Create interpreter with PSInterpreter. Run PSInterpreter.interpret to get output array and dictionary

# Arguments
rf_center (float): RF center (local oscillator frequency) in Hz.

rf_amp_max (float): Default 5e+3 -- System RF amplitude max in Hz.

grad_max (float): Default 1e+6 -- System gradient max in Hz/m.

clk_t (float): Default 1/122.88 -- System clock period in us.

tx_t (float): Default 123/122.88 -- Transmit raster period in us. Will be overwritten if the PulSeq file includes a "tx_t" in the definitions

grad_t (float): Default 1229/122.88 -- Gradient raster period in us. Will be overwritten if the PulSeq file includes a "grad_t" in the definitions

# Outputs
dict: tuple of numpy.ndarray time and update arrays, with variable name keys

dict: parameter dictionary containing raster times, readout numbers, and any file-defined variables

