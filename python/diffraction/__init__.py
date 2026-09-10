"""Sound occlusion by a barrier, via time-domain edge diffraction.

An implementation of the Biot-Tolstoy-Medwin edge-diffraction integral in the
form given by Svensson, Fred and Vanderkooy (1999): a secondary-source line
integral along the edge, evaluated in the time domain at 48 kHz.
"""

__version__ = "0.1.0"

SPEED_OF_SOUND = 343.0
SAMPLE_RATE = 48000.0
