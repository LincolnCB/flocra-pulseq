# -*- coding: utf-8 -*-
# pulseq_assembler.py
# Written by Lincoln Craven-Brightman

import pdb # Debugging
import numpy as np
import logging # For errors
import struct
import io

ROUNDING = 0
# TODO: Split up gradients

class PSAssembler:
    """
    Assembler object that can assemble a PulSeq file into OCRA machine code. Run PSAssembler.assemble to compile a .seq file into OCRA machine code

    Attributes:
        tx_arr (float): Transmit data bytes
        grad_arr (float): List of grad bytes
        readout_number (int): Expected number of readouts
    """

    def __init__(self, rf_center=3e+6, rf_amp_max=5e+3, grad_max=1e+7,
                 clk_t=7e-3, tx_t=1.001, grad_t=10.003,
                 pulseq_t_match=False, ps_tx_t=1, ps_grad_t=10):
        """
        Create PSAssembler object with system parameters.

        Args:
            rf_center (int): RF center (local oscillator frequency) in Hz.
            rf_amp_max (int): Default 5e+3 -- System RF amplitude max in Hz.
            grad_max (int): Default 1e+6 -- System gradient max in Hz/m.
            clk_t (float): Default 7e-3 -- System clock period in us.
            tx_t (float): Default 1.001 -- Transmit raster period in us.
            grad_t (float): Default 10.003 -- Gradient raster period in us.
            pulseq_t_match (bool): Default False -- If PulSeq file transmit and gradient raster times match OCRA transmit and raster times.
            ps_tx_t (float): Default 1 -- PulSeq transmit raster period in us. Used only if pulseq_t_match is False.
            ps_grad_t (float): Default 10 -- PulSeq gradient raster period in us. Used only if pulseq_t_match is False.
        """
        # Logging
        self._logger = logging.getLogger()
        logging.basicConfig(filename = 'psassembler.log', filemode = 'w', level = logging.DEBUG)

        # PulSeq dictionary storage
        self._blocks = {}
        self._rf_events = {}
        self._grad_events = {}
        self._adc_events = {}
        self._delay_events = {}
        self._shapes = {}

        # Interpreter for section names in .seq file
        self._pulseq_keys = {
            '[VERSION]' : self._read_temp, # Unused
            '[DEFINITIONS]' : self._read_defs,
            '[BLOCKS]' : self._read_blocks,
            '[RF]' : self._read_rf_events,
            '[GRADIENTS]' : self._read_grad_events,
            '[TRAP]' : self._read_trap_events,
            '[ADC]' : self._read_adc_events,
            '[DELAYS]' : self._read_delay_events,
            '[EXTENSIONS]' : self._read_temp, # Unused
            '[SHAPES]' : self._read_shapes
        }

        self._clk_t = clk_t # Instruction clock period in us
        self._tx_div = int(tx_t / self._clk_t + ROUNDING) # Clock cycles per tx
        self._tx_t = tx_t # Transmit sample period in us
        self._warning_if(self._tx_div * self._clk_t != tx_t, 
            f'tx_t ({tx_t}) rounded to {self._tx_t}, multiple of clk_t ({clk_t})')
        self._grad_div = int(grad_t / self._clk_t + ROUNDING) # Clock cycles per grad
        self._grad_t = clk_t * self._grad_div # Gradient sample period in us
        self._warning_if(self._grad_div * self._clk_t != grad_t, 
            f'grad_t ({(grad_t)}) rounded to {self._grad_t}, multiple of clk_t ({clk_t})')
        self._rx_div = None
        self._rx_t = None

        if not pulseq_t_match:
            self._ps_tx_t = ps_tx_t # us
            self._ps_grad_t = ps_grad_t # us
        else:
            self._ps_tx_t = tx_t # us
            self._ps_grad_t = grad_t # us

        self._rf_center = rf_center # Hz
        self._rf_amp_max = rf_amp_max # Hz
        self._grad_max = grad_max # Hz/m

        self._tx_offsets = {} # Tx word index (32-bit) by Tx ID
        self._tx_delays = {} # us
        self._tx_durations = {} # us
        self._grad_offsets = {} # Gradient word index (3 concurrent 32-bit) by Gx, Gy, Gz ID combo
        self._grad_delays = {} # us
        self._grad_durations = {} # us
        self._definitions = {}

        self.tx_arr = np.zeros(0, dtype=np.complex_)
        self.grad_arr = np.zeros((3, 0)) # x, y, z
        self.out_arr = np.zeros(0, dtype=np.complex_)
        self.readout_number = 0
        self.is_assembled = False

    # Wrapper for full assembly
    def assemble(self, pulseq_file, byte_format=True):
        """
        Assemble OCRA machine code from PulSeq .seq file

        Args:
            pulseq_file (str): PulSeq file to assemble from
            byte_format (bool): Default True -- Return transmit and gradient data in bytes, rather than numpy.ndarray
        
        Returns:
            tuple: Transmit data (bytes or numpy.ndarray); list of gradient data (list) (bytes or numpy.ndarray);
                 command bytes (bytes); dictionary of final outputs (dict)
        """
        self._logger.info(f'Assembling ' + pulseq_file)
        if self.is_assembled:
            self._logger.info('Overwriting old sequence...')
        
        self._read_pulseq(pulseq_file)
        self._compile_tx_data()
        self._compile_grad_data()
        self.out_arr = self._stream_all_blocks()
        self.is_assembled = True
        output_dict = {'readout_number' : self.readout_number, 'tx_t' : self._tx_t, 'rx_t' : self._rx_t}
        for key, value in self._definitions.items():
            output_dict[key] = value
        return (self.out_arr, output_dict)
        
    # Open file and read in all sections into class storage
    def _read_pulseq(self, pulseq_file):
        """
        Read PulSeq file into object dict memory

        Args:
            pulseq_file (str): PulSeq file to assemble from
        """
        # Open file
        with open(pulseq_file) as f:
            self._logger.info('Opening PulSeq file...')
            line = '\n'
            next_line = ''

            while True:
                if not next_line: 
                    line = f.readline()
                else: 
                    line = next_line
                    next_line = ''
                if line == '': break
                key = self._simplify(line)
                if key in self._pulseq_keys:
                    next_line = self._pulseq_keys[key](f)

        # Check that all ids are valid
        self._logger.info('Validating ids...')
        var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
        var_dicts = [self._delay_events, self._rf_events, self._grad_events, self._grad_events, self._grad_events, self._adc_events, {}]
        for block in self._blocks.values():
            for i in range(len(var_names)):
                id_n = block[var_names[i]]
                self._error_if(id_n != 0 and id_n not in var_dicts[i], f'Invalid {var_names[i]} id: {id_n}')
        for rf in self._rf_events.values():
            self._error_if(rf['mag_id'] not in self._shapes, f'Invalid magnitude shape id: {rf["mag_id"]}')
            self._error_if(rf['phase_id'] not in self._shapes, f'Invalid phase shape id: {rf["phase_id"]}')
        for grad in self._grad_events.values():
            if len(grad) == 3:
                self._error_if(grad['shape_id'] not in self._shapes, f'Invalid grad shape id: {grad["shape_id"]}')
        self._logger.info('Valid ids')

        # Check that all delays are multiples of clk_t
        for events in [self._blocks.values(), self._rf_events.values(), self._grad_events.values(), 
                        self._adc_events.values()]:
            for event in events:
                self._warning_if(int(event['delay'] / self._clk_t + ROUNDING) * self._clk_t != event['delay'],
                    f'Delay is not a multiple of clk_t, rounding')
        for delay in self._delay_events.values():
            self._warning_if(int(delay / self._clk_t + ROUNDING) * self._clk_t != delay,
                f'Delay is not a multiple of clk_t, rounding')
        
        # Check that RF/ADC (TX/RX) only have one frequency offset -- can't be set within one file.
        freq = None
        base_id = None
        base_str = None
        for rf_id, rf in self._rf_events.items():
            if freq is None:
                freq = rf['freq']
                base_id = rf_id
                base_str = 'RF'
            self._error_if(rf['freq'] != freq, f"Frequency offset of RF event {rf_id} ({rf['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
        for adc_id, adc in self._adc_events.items():
            if freq is None:
                freq = adc['freq']
                base_id = adc_id
                base_str = 'ADC'
            self._error_if(adc['freq'] != freq, f"Frequency offset of ADC event {adc_id} ({adc['freq']}) doesn't match that of {base_str} event {base_id} ({freq})")
        if freq is not None and freq != 0:
            self._rf_center += freq
            self._logger.info(f'Adding freq offset {freq} Hz. New center / linear oscillator frequency: {self._rf_center}')

        # Check that ADC has constant dwell time
        dwell = None
        for adc_id, adc in self._adc_events.items():
            if dwell is None:
                dwell = adc['dwell']/1000
                base_id = adc_id
            self._error_if(adc['dwell']/1000 != dwell, f"Dwell time of ADC event {adc_id} ({adc['dwell']}) doesn't match that of ADC event {base_id} ({dwell})")
        if dwell is not None:
            self._rx_div = np.round(dwell / self._clk_t).astype(int)
            self._rx_t = self._clk_t * self._rx_div
            self._warning_if(self._rx_div * self._clk_t != dwell, 
                f'Dwell time ({dwell}) rounded to {self._rx_t}, multiple of clk_t ({self._clk_t})')
        
        self._logger.info('PulSeq file loaded')
    
    # Compilation into data formats
    #region

    # Compile tx events into bytes
    def _compile_tx_data(self):
        """
        Compile transmit data from object dict memory into bytes
        """

        self._logger.info('Compiling Tx data...')
        tx_data = []
        curr_offset = 0

        # Process each rf event
        for tx_id, tx in self._rf_events.items():
            # Collect mag/phase shapes
            mag_shape = self._shapes[tx['mag_id']]
            phase_shape = self._shapes[tx['phase_id']]
            if len(mag_shape) != len(phase_shape):
                self._logger.warning(f'Tx envelope of RF event {tx_id} has mismatched magnitude and phase information,'
                                    ' the last entry of the shorter will be extended')

            # Array length, unitless -- extends shorter of phase/mag shape to length of longer                     
            pulse_len = int((max(len(mag_shape), len(phase_shape)) - 1) * self._ps_tx_t / self._tx_t) + 1 # unitless
            
            # Interpolate values on falling edge (and extend past end of shorter, if needed)
            x = np.flip(np.linspace(pulse_len * self._tx_t, 0, num=pulse_len, endpoint=False)) # us
            mag_x_ps = np.flip(np.linspace(len(mag_shape)* self._ps_tx_t, 0, num=len(mag_shape), endpoint=False))
            phase_x_ps = np.flip(np.linspace(len(phase_shape)* self._ps_tx_t, 0, num=len(phase_shape), endpoint=False))
            mag_interp = np.interp(x, mag_x_ps, mag_shape) * tx['amp'] / self._rf_amp_max
            phase_interp = np.interp(x, phase_x_ps, phase_shape) * 2 * np.pi

            tx_env = np.zeros(pulse_len, dtype=np.complex64)

            # Convert to complex tx envelope
            tx_env = np.exp((phase_interp + tx['phase']) * 1j) * mag_interp
            
            if np.any(np.abs(tx_env) > 1.0):
                self._logger.warning(f'Magnitude of RF event {tx_id} was too large, 16-bit signed overflow will occur')
            
            # Concatenate tx data and track offsets
            tx_data.extend(tx_env.tolist())
            self._tx_offsets[tx_id] = curr_offset
            self._tx_durations[tx_id] = pulse_len * self._tx_t
            self._tx_delays[tx_id] = tx['delay']
            curr_offset += pulse_len

        # Save TX array for external use
        self.tx_arr = np.array(tx_data)
        self._logger.info('Tx data compiled')

    # Compile grad events into bytes
    def _compile_grad_data(self):
        """
        Compile gradient events from object dict memory into bytes
        """
        # Prep grad data
        self._create_helper_shapes()

        self._logger.info('Compiling gradient data...')
        grad_data = [[], [], []]
        curr_offset = 0

        # Process each block (all gradients play out at once, so different xyz combinations are distinct)
        for block in self._blocks.values():
            grad_ids = (block['gx'], block['gy'], block['gz'])

            # Skip if all off or a repeat combination
            if grad_ids[0] == 0 and grad_ids[1] == 0 and grad_ids[2] == 0: continue
            if (grad_ids) in self._grad_offsets: continue

            # Collect grad events and shapes
            grads = [self._grad_events[grad_ids[i]] for i in range(3)]
            grad_shapes = [self._shapes[grads[i]['shape_id']] for i in range(3)]

            # Remove time when all are off
            grad_delays = [grad['delay'] for grad in grads]
            min_delay = min(grad_delays)
            grad_delay_lens = [int((delay - min_delay) / self._ps_grad_t + ROUNDING) if delay != np.inf else 0 for delay in grad_delays]

            # Array lengths (unitless)
            grad_ps_len = max([len(grad_shapes[i]) + grad_delay_lens[i] for i in range(3)])
            grad_len = int(grad_ps_len * self._ps_grad_t / self._grad_t + ROUNDING)

            # Falling edge time arrays for interpolation
            duration = grad_ps_len * self._ps_grad_t
            x_ps = np.flip(np.linspace(duration + self._ps_grad_t, 0, num=grad_ps_len + 2)) # Add a zero on either end
            x = np.flip(np.linspace(duration, 0, num=grad_len, endpoint=False))
            
            # Interpolate, scale, and concatenate grad data
            for i in range(3):
                grad_ps = np.zeros(grad_ps_len + 2)
                grad_ps[grad_delay_lens[i] + 1 : grad_delay_lens[i] + len(grad_shapes[i]) + 1] = np.array(grad_shapes[i])
                gr = np.interp(x, x_ps, grads[i]['amp'] * grad_ps) / self._grad_max

                grad_data[i].extend(gr.tolist())
                if np.any(np.abs(gr) > 1.0):
                    self._logger.warning(f'Magnitude of gradient event {grad_ids[i]} was too large, 16-bit signed overflow will occur')

            # Track offsets for concatenated grad events
            self._grad_offsets[grad_ids] = curr_offset
            self._grad_durations[grad_ids] = grad_len * self._grad_t
            self._grad_delays[grad_ids] = min_delay
            curr_offset += grad_len

        # store floating-point arrays
        self.grad_arr = np.array([np.array(channel) for channel in grad_data])
        self._logger.info('Gradient data compiled')

    # Create shapes to convert trapezoids into the same format as gradients, and add a zero shape for when not all of X, Y, Z are on at once. 
    def _create_helper_shapes(self):
        """
        Creates rastered shapes for trapezoid events for encoding into gradient bytes, and creates a zero shape for when not all of X, Y, Z are on at once.
        """
        self._logger.info('Processing trapezoids...')
        # Append helper shapes on top of existing shapes
        max_id = 0
        for shape_id in self._shapes:
            if shape_id > max_id: max_id = shape_id

        # Append zero shape first
        max_id += 1
        self._grad_events[0] = {'amp': 0, 'shape_id': max_id, 'delay': np.inf}
        self._shapes[max_id] = np.zeros(0)
        
        # Create and append new trap shapes, and convert trap into standard grad events
        for grad_id, grad in self._grad_events.items():
            if len(grad) == 5:
                rise = np.flip(np.linspace(1, 0, num=int(grad['rise'] / self._ps_grad_t + ROUNDING), endpoint=False))
                flat = np.ones(int(grad['flat'] / self._ps_grad_t + ROUNDING))
                fall = np.flip(np.linspace(0, 1, num=int(grad['fall'] / self._ps_grad_t + ROUNDING), endpoint=False))
                shape = np.concatenate((rise, flat, fall))
                
                max_id += 1
                self._shapes[max_id] = shape
                self._grad_events[grad_id] = {'amp': grad['amp'], 'shape_id': max_id, 'delay': grad['delay']}
        self._logger.info('Trapezoids processed')
        return

    # Encode all blocks
    def _stream_all_blocks(self):
        """
        Encode all blocks into sequential gate changes.

        Returns:
            Aligned lists of durations (us), gates, TX and GRAD offsets for sequential instructions. 
        """
        # Encode all blocks
        out_arrs = []
        start = 0
        for block_id in self._blocks.keys():
            out_arr, duration = self._stream_block(block_id)
            out_arr[:, 0] += start
            out_arrs.append(out_arr)
            start += duration

        # Zero gates at the end
        ending = np.zeros((1, out_arrs[0].shape[1]))
        ending[:, 0] += start
        out_arrs.append(ending)

        return (np.concatenate(out_arrs, axis=0))

    # Convert individual block into PR commands (duration, gates), TX offset, and GRAD offset
    def _stream_block(self, block_id):
        """
        Encode block into sequential gate changes to be compiled into byte instructions

        Args:
            block_id (int): Block id key for block in object dict memory to be encoded
        
        Returns:
            tuple: PR durations (list) (int); transmit address changes for each PR, -1 if no change (np.ndarray);
                gradient address changes for each time, -1 if no change (np.ndarray)
        """
        block = self._blocks[block_id]
        
        # Determine important times in us (when gates change)
        if block['delay'] != 0:
            delay = self._delay_events[block['delay']]
        else:
            delay = 0
        tx_start = tx_end = grad_start = grad_end = rx_start = rx_end = 0
        tx_len = grad_len = rx_len = 0
        tx_id = block['rf']
        grad_ids = (block['gx'], block['gy'], block['gz'])
        adc_id = block['adc']
        if tx_id != 0: # rf timing
            tx_start = self._tx_delays[tx_id]
            tx_end = self._tx_durations[tx_id] + tx_start
        if grad_ids != (0, 0, 0): # grad timing
            grad_start = self._grad_delays[grad_ids]
            grad_end = self._grad_durations[grad_ids] + grad_start
        if adc_id:
            adc = self._adc_events[adc_id]
            rx_start = adc['delay']
            rx_end = self._rx_t * adc['num'] + adc['delay']
            self.readout_number += adc['num']

        # Confirm min delay from delay event is met. 
        duration = max([tx_start, tx_end, grad_start, grad_end, rx_start, rx_end, 0, delay])

        times_list = []
        if tx_start < tx_end:
            tx_len = int((tx_end - tx_start) / self._tx_t)
            times_list.extend(np.linspace(tx_start, tx_end, num=tx_len, endpoint=False))
        if grad_start < grad_end:
            grad_len = int((grad_end - grad_start) / self._grad_t)
            times_list.extend(np.linspace(grad_start, grad_end, num=grad_len, endpoint=False))
        if rx_start < rx_end:
            rx_len = int((rx_end - rx_start) / self._rx_t)
            times_list.extend(np.linspace(rx_start, rx_end, num=rx_len, endpoint=False))

        times = np.sort(np.unique(np.asarray(times_list)))

        tx_vals = np.zeros((times.shape[0], 1), dtype=np.complex_)
        grad_vals = np.zeros((times.shape[0], 3), dtype=np.float_)
        rx_gate = np.zeros((times.shape[0], 1), dtype=np.bool_)

        tx_change = (times >= tx_start) * (times < tx_end) * \
            (np.abs(np.round((times - tx_start) / self._tx_t) - (times - tx_start) / self._tx_t) < 1e-13)
        grad_change = (times >= grad_start) * (times < grad_end) * \
            (np.abs(np.round((times - grad_start) / self._grad_t) - (times - grad_start) / self._grad_t) < 1e-13)
        rx_change = (times >= rx_start) * (times < rx_end) * ((times - rx_start) % self._rx_t == 0)

        if tx_id != 0:
            tx_vals[tx_change] = self.tx_arr[self._tx_offsets[tx_id]:self._tx_offsets[tx_id] + tx_len].reshape(-1, 1)
        if grad_ids != (0, 0, 0):
            grad_vals[grad_change, :] = self.grad_arr[:, self._grad_offsets[grad_ids]:self._grad_offsets[grad_ids] + grad_len].T
        rx_gate[rx_change] = 1

        for ind in range(1, times.shape[0]):
            if not tx_change[ind]:
                tx_vals[ind] = tx_vals[ind - 1]
            if not grad_change[ind]:
                grad_vals[ind, :] = grad_vals[ind - 1, :]
            if not rx_change[ind]:
                rx_gate[ind] = rx_gate[ind - 1]

        out_arr = np.concatenate((times.reshape(-1, 1), tx_vals, grad_vals, rx_gate), axis=1)

        # Return durations for each PR and leading edge values
        return (out_arr, duration)
    #endregion

    # Helper functions for reading sections
    #region

    # [BLOCKS] <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>
    def _read_blocks(self, f):
        """
        Read BLOCKS (event block) section in PulSeq file f to object dict memory.
        Event blocks are formatted like: <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('delay', 'rf', 'gx', 'gy', 'gz', 'adc', 'ext')
        rline = ''
        line = ''
        self._logger.info('Blocks: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 8: # <id> <delay> <rf> <gx> <gy> <gz> <adc> <ext>
                data_line = [int(x) for x in tmp]
                self._warning_if(data_line[0] in self._blocks, f'Repeat block ID {data_line[0]}, overwriting')
                self._blocks[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
            elif len(tmp) == 7: # Spec allows extension ID not included, add it in as 0
                data_line = [int(x) for x in tmp]
                data_line.append(0)
                self._warning_if(data_line[0] in self._blocks, f'Repeat block ID {data_line[0]}, overwriting')
                self._blocks[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
        
        if len(self._blocks) == 0: self._logger.error('Zero blocks read, nonzero blocks needed')
        assert len(self._blocks) > 0, 'Zero blocks read, nonzero blocks needed'
        self._logger.info('Blocks: Complete')

        return rline

    # [RF] <id> <amp> <mag_id> <phase_id> <delay> <freq> <phase>
    def _read_rf_events(self, f):
        """
        Read RF (RF event) section in PulSeq file f to object dict memory.
        RF events are formatted like: <id> <amp> <mag_id> <phase_id> <delay> <freq> <phase>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'mag_id', 'phase_id', 'delay', 'freq', 'phase')
        rline = ''
        line = ''
        self._logger.info('RF: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 7: # <id> <amp> <mag id> <phase id> <delay> <freq> <phase>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5]), float(tmp[6])]
                self._warning_if(data_line[0] in self._rf_events, f'Repeat RF ID {data_line[0]}, overwriting')
                self._rf_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('RF: Complete')

        return rline

    # [GRADIENTS] <id> <amp> <shape_id> <delay>
    def _read_grad_events(self, f):
        """
        Read GRADIENTS (gradient event) section in PulSeq file f to object dict memory.
        Gradient events are formatted like: <id> <amp> <shape_id> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'shape_id', 'delay')
        rline = ''
        line = ''
        self._logger.info('Gradients: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 4: # GRAD <id> <amp> <shape id> <delay>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3])]
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in GRADIENTS, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
            elif len(tmp) == 3: # GRAD <id> <amp> <shape id> NO DELAY
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2])]
                data_line.append(0)
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]}, in GRADIENTS, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('Gradients: Complete')

        return rline

    # [TRAP] <id> <amp> <rise> <flat> <fall> <delay>
    def _read_trap_events(self, f):
        """
        Read TRAP (trapezoid gradient event) section in PulSeq file f to object dict memory.
        Trapezoid gradient events are formatted like: <id> <amp> <rise> <flat> <fall> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('amp', 'rise', 'flat', 'fall', 'delay')
        rline = ''
        line = ''
        self._logger.info('Trapezoids: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 6: # TRAP <id> <amp> <rise> <flat> <fall> <delay>
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), float(tmp[5])]
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}
            elif len(tmp) == 5: # TRAP <id> <amp> <rise> <flat> <fall> NO DELAY
                data_line = [int(tmp[0]), float(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4])]
                data_line.append(0)
                self._warning_if(data_line[0] in self._grad_events, f'Repeat gradient ID {data_line[0]} in TRAP, overwriting')
                self._grad_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('Trapezoids: Complete')

        return rline

    # [ADC] <id> <num> <dwell> <delay> <freq> <phase>
    def _read_adc_events(self, f):
        """
        Read ADC (ADC/readout event) section in PulSeq file f to object dict memory.
        ADC events are formatted like: <id> <num> <dwell> <delay> <freq> <phase>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        var_names = ('num', 'dwell', 'delay', 'freq', 'phase')
        rline = ''
        line = ''
        self._logger.info('ADC: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 6:
                data_line = [int(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3]), float(tmp[4]), float(tmp[5])]
                self._adc_events[data_line[0]] = {var_names[i] : data_line[i+1] for i in range(len(var_names))}

        self._logger.info('ADC: Complete')

        return rline

    # [DELAY] <id> <delay> -> single value output
    def _read_delay_events(self, f):
        """
        Read DELAY (delay event) section in PulSeq file f to object dict memory (stored as a single value, not a dict).
        Delay events are formatted like: <id> <delay>

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Delay: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 2:
                data_line = [int(x) for x in tmp]
                self._warning_if(data_line[0] in self._delay_events, f'Repeat delay ID {data_line[0]}, overwriting')
                self._delay_events[data_line[0]] = data_line[1] # Single value, delay

        self._logger.info('Delay: Complete')

        return rline

    # [SHAPES] list of entries, normalized between 0 and 1
    def _read_shapes(self, f):
        """
        Read SHAPES (rastered shapes) section in PulSeq file f to object dict memory.
        Shapes are formatted with two header lines, followed by lines of single data points in compressed pulseq shape format

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Shapes: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break
            if len(rline.split()) == 2 and rline.split()[0].lower() == 'shape_id':
                shape_id = int(rline.split()[1])
                n = int(self._simplify(f.readline()).split()[1])
                self._warning_if(shape_id in self._shapes, f'Repeat shape ID {shape_id}, overwriting')
                self._shapes[shape_id] = np.zeros(n)
                i = 0
                prev = -2
                x = 0
                while i < n:
                    dx = float(self._simplify(f.readline()))
                    x += dx
                    self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x}, outside of [0, 1], rounding')
                    if x > 1:
                        x = 1
                    elif x < 0:
                        x = 0
                    self._shapes[shape_id][i] = x
                    if dx == prev:
                        r = int(self._simplify(f.readline()))
                        for _ in range(0, r):
                            i += 1
                            x += dx
                            self._warning_if(x > 1 or x < 0, f'Shape {shape_id} entry {i} is {x}, outside of [0, 1], rounding')
                            if x > 1:
                                x = 1
                            elif x < 0:
                                x = 0
                            self._shapes[shape_id][i] = x
                    i += 1
                    prev = dx

        self._logger.info('Shapes: Complete')

        return rline

    # [DEFINITIONS] <varname> <value>
    def _read_defs(self, f):
        """
        Read through DEFINITIONS section in PulSeq file f.

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('Definitions: Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break

            tmp = rline.split()
            if len(tmp) == 2:
                varname, value = rline.split()
                try:
                    value = float(value)
                except:
                    pass
                self._definitions[varname] = value
                self._logger.debug(f'Read in {varname}')

        self._logger.info('Definitions: Complete')

        return rline

    # Unused headers
    def _read_temp(self, f):
        """
        Read through any unused section in PulSeq file f.

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Raw next line in file after section ends
        """
        rline = ''
        line = ''
        self._logger.info('(Unused): Reading...')
        while True:
            line = f.readline()
            rline = self._simplify(line)
            if line == '' or rline in self._pulseq_keys: break
            self._logger.debug('Unused line')

        self._logger.info('(Unused): Complete')

        return rline

    # Simplify lines read from pulseq -- remove comments, trailing \n, trailing whitespace, commas
    def _simplify(self, line):
        """
        Simplify raw line to space-separated values

        Args:
            f (_io.TextIOWrapper): File pointer to read from

        Returns:
            str: Simplified string
        """

        # Find and remove comments, comma
        comment_index = line.find('#')
        if comment_index >= 0:
            line = line[:comment_index]
        
        return line.rstrip('\n').strip().replace(',','')
    
    #endregion
    
    # Error and warnings
    #region
    # For crashing and logging errors (may change behavior)
    def _error_if(self, err_condition, message):
        """
        Throw an error (currently using assert) and log if error condition is met

        Args:
            err_condition (bool): Condition on which to throw error
            message (str): Message to accompany error in log. 
        """
        if err_condition: self._logger.error(message)
        assert not err_condition, (message)

    # For warnings without crashing
    def _warning_if(self, warn_condition, message):
        """
        Print warning and log if error condition is met

        Args:
            warn_condition (bool): Condition on which to warn
            message (str): Message to accompany warning in log. 
        """
        if warn_condition: self._logger.warning(message)
    #endregion

# Sample usage
if __name__ == '__main__':
    ps = PSAssembler()
    inp_file = '../ocra-pulseq/test_files/test_loopback.seq'
    out_arr, params = ps.assemble(inp_file)
    print(out_arr[:, 0])
    print("Completed successfully")