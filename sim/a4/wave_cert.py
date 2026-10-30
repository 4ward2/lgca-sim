"""Finite local wave obligations, independently recomputed by replay.

This gate describes explicit wave reached bits, not a learned AC certificate.
Learned AC remains disabled unless its complete domain obligations are supplied.
"""
import itertools
from .protocol import digest

def wave_certificate():
    obligations=[]
    for wall,source,reached,*neighbors in itertools.product((0,1),repeat=7):
        expected=int(not wall and (source or reached or any(neighbors)))
        obligations.append([wall,source,reached,*neighbors,expected])
    return dict(kind='explicit-wave-oracle',V=2,obligations=obligations,sha256=digest(obligations),learned_ac_verdict='UNKNOWN')

def replay(payload):
    expected=wave_certificate()
    if payload!=expected: raise ValueError('missing, duplicated or forged wave obligations')
    return True
