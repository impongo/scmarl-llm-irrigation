from casadi import *
from richards_equation_ekf import *
from temporal_spatial_ekf import *
totalDepth, axialNodes, axialDistance, totalNodes = spatialVariables_1D()
def evaluate_jacobian(head, irrigAmount, cropCoeff, refEvap, laiFactor, soilPars):
	head_sym = SX.sym('h', totalNodes)
	irrigRate_sym = SX.sym('u', 1)
	cropCoeff_sym = SX.sym('c', 1)
	refEvap_sym = SX.sym('e', 1)
	laiFactor_sym = SX.sym('l', 1)
	symbolicModel = RichardsPolar_1D(head_sym, irrigRate_sym, cropCoeff_sym, refEvap_sym, laiFactor_sym, soilPars)
	jacHead = jacobian(symbolicModel, head)
	jacHeadFun = Function('jacHeadFun', [head_sym, irrigRate_sym, cropCoeff_sym, refEvap_sym, laiFactor_sym],[jacobianHead])

	A_continuous = jacHeadFun(head, irrigAmount, cropCoeff, refEvap, laiFactor)
	return A_continuous