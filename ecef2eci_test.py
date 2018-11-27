from sympy import *
import numpy as np

# %% Derivation Function:
# clear
ws, dt, Peci1, Peci2, Peci3, Veci1, Veci2, Veci3, Aeci1, Aeci2, Aeci3, Jeci1, Jeci2, Jeci3\
    = symbols('ws dt Peci1 Peci2 Peci3 Veci1 Veci2 Veci3 Aeci1 Aeci2 Aeci3 Jeci1 Jeci2 Jeci3')

init_printing(use_unicode=True)

Peci = Matrix([Peci1, Peci2, Peci3])
Veci = Matrix([Veci1, Veci2, Veci3])
Aeci = Matrix([Aeci1, Aeci2, Aeci3])
Jeci = Matrix([Jeci1, Jeci2, Jeci3])

Cwdt = cos(ws*dt)
Swdt = sin(ws*dt)
Te2i = Matrix([[Cwdt, -Swdt, 0], [Swdt, Cwdt, 0], [0, 0, 1]])

Ti2e = Te2i.T

zvv = Matrix([0, 0, ws])

Pecef = simplify(Ti2e.multiply(Peci))
Vecef = simplify(Ti2e.multiply(Veci) + zvv.cross(Pecef))
Aecef = simplify(Ti2e.multiply(Aeci) + 2*zvv.cross(Vecef) + zvv.cross(zvv.cross(Pecef)))
Jecef = simplify(Ti2e.multiply(Jeci) + 2*zvv.cross(Aecef) + zvv.cross(zvv.cross(zvv.cross(Pecef))))

state_ecef = Matrix([Pecef, Vecef, Aecef, Jecef])

state_eci1 = Matrix([Peci0, Veci0, Aeci0, Jeci0])

ws, dt, Pecef1, Pecef2, Pecef3, Vecef1, Vecef2, Vecef3, Aecef1, Aecef2, Aecef3, Jecef1, Jecef2, Jecef3\
    = symbols('ws dt Pecef1 Pecef2 Pecef3 Vecef1 Vecef2 Vecef3 Aecef1 Aecef2 Aecef3 Jecef1 Jecef2 Jecef3')

init_printing(use_unicode=True)

Pecef = Matrix([Pecef1, Pecef2, Pecef3])
Vecef = Matrix([Vecef1, Vecef2, Vecef3])
Aecef = Matrix([Aecef1, Aecef2, Aecef3])
Jecef = Matrix([Jecef1, Jecef2, Jecef3])

Cwdt = cos(ws*dt)
Swdt = sin(ws*dt)
Te2i = Matrix([[Cwdt, -Swdt, 0], [Swdt, Cwdt, 0], [0, 0, 1]])

zvv = Matrix([0, 0, ws])

Peci = Te2i.multiply(Pecef)
Veci = Te2i.multiply(Vecef) - zvv.cross(Pecef)
Aeci = Te2i.multiply(Aecef) - 2*zvv.cross(Vecef) - zvv.cross(zvv.cross(Pecef))
Jeci = Te2i.multiply(Jecef) - 2*zvv.cross(Aecef) - zvv.cross(zvv.cross(zvv.cross(Pecef)))

state_eci2 = Matrix([Peci, Veci, Aeci, Jeci])

# Pecef=Ti2e*[Peci1 Peci2 Peci3]'
#
# Vecef=Ti2e*[Veci1; Veci2; Veci3]-cross([0;0;w],Pecef)
# Aecef=Ti2e*[Aeci1 Aeci2 Aeci3]'-2*cross([0;0;w],Vecef)-cross([0;0;w],cross([0;0;w],Pecef))
#
# Peci=simplify(expand(Te2i*Pecef))
# Veci=simplify(expand(Te2i*Vecef+cross([0;0;w],Peci)))
# Aeci=simplify(expand(Te2i*Aecef+2*cross([0;0;w],Te2i*Vecef)+cross([0;0;w],cross([0;0;w],Peci))))