# flocra-pulseq
Pulseq interpreter for vnegnev's flow-based OCRA

# Usage:
Create interpreter with PSInterpreter. Run PSInterpreter.compile to get output array and dictionary

# Arguments
rf_center (int): RF center (local oscillator frequency) in Hz.

rf_amp_max (int): Default 5e+3 -- System RF amplitude max in Hz.

grad_max (int): Default 1e+6 -- System gradient max in Hz/m.

clk_t (float): Default 7e-3 -- System clock period in us.

tx_t (float): Default 1.001 -- Transmit raster period in us.

grad_t (float): Default 10.003 -- Gradient raster period in us.

pulseq_t_match (bool): Default False -- If PulSeq file transmit and gradient raster times match FLOCRA transmit and raster times.

ps_tx_t (float): Default 1 -- PulSeq transmit raster period in us. Used only if pulseq_t_match is False.

ps_grad_t (float): Default 10 -- PulSeq gradient raster period in us. Used only if pulseq_t_match is False.

# Outputs
Complex (updates)x(variables) array (time, tx, gx, gy, gz, rx_gate) of sequence

Dictionary of variables used in interpreter or passed through pulseq file. Check 'output_labels' for variable ordering of sequence array

