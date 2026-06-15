#!/usr/bin/env python3
"""Generate tiny numpy-produced fixtures for the neuralnumpy.pas parity test.

Run with the reusable venv:  /home/bpsa/x/bin/python tools/numpy_crosscheck_fixture.py

Writes (under tests/fixtures/):
  numpy_crosscheck_f32.npy : np.arange(12, dtype=f4).reshape(3, 4)
  numpy_crosscheck.npz     : {'ramp': arange(6,f4), 'half': arange(6,f2)*0.5}

The .npz is uncompressed (np.savez). Both files are a few hundred bytes so
they are safe to commit as parity fixtures. The Pascal reader is asserted
against these in tests/TestNeuralNumpy.TestNumpyCrossCheckFixture.

This script can also VERIFY that a Pascal-written file np.load's correctly:
  python tools/numpy_crosscheck_fixture.py --verify <file.npy|file.npz>
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, '..', 'tests', 'fixtures')


def generate():
    os.makedirs(FIX, exist_ok=True)
    a = np.arange(12, dtype='<f4').reshape(3, 4)
    np.save(os.path.join(FIX, 'numpy_crosscheck_f32.npy'), a)
    ramp = np.arange(6, dtype='<f4')
    half = (np.arange(6, dtype='<f2') * 0.5).astype('<f2')
    np.savez(os.path.join(FIX, 'numpy_crosscheck.npz'), ramp=ramp, half=half)
    print('wrote numpy_crosscheck_f32.npy and numpy_crosscheck.npz to', FIX)


def verify(path):
    if path.endswith('.npz'):
        with np.load(path) as z:
            for k in z.files:
                print(f'{k}: shape={z[k].shape} dtype={z[k].dtype}')
                print(z[k])
    else:
        a = np.load(path)
        print(f'shape={a.shape} dtype={a.dtype}')
        print(a)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--verify':
        verify(sys.argv[2])
    else:
        generate()
