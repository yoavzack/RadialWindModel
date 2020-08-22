import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import scipy.optimize as opt
from matplotlib.widgets import Slider, Button, RadioButtons

from decimal import Decimal, localcontext

mpl.rcParams['grid.linestyle'] = "--"
# plt.style.use('dark_background')
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# numerical things--------------------------------------------------------------------
# Define constants
z0			= 500.0
G			= 1.0
M_BH		= 1.0
gam			= 5.0/3.0
k			= 4.0
r_in		= 10.0
r_out		= 200.0
th_wmin		= np.arctan(r_in/z0)
th_jmax		= 0.577
r_L			= 2.25 # rL=c/Omega
L_j			= np.pi*r_L**2
p_w0		= 1e-10
b_w			= 0.605

# constants in question???
u_win = 1.0

# Technical constants for python
plotRange = 1000.0







# Slider Parameters ---------------------------------------------------------------
def update(val):
	# get parameter

	# initial condition
	z_c = z0*np.tan(th_wmin)/(np.tan(th_jmax)-np.tan(th_wmin))
	VARVAR0 = [z_c*np.tan(th_jmax),z_c*np.tan(th_jmax),np.tan(th_jmax)]

	# solve ODE
	n = 101
	z = np.linspace(0,plotRange,n)
	sol, infodict = odeint(model,VARVAR0,z,full_output=1)
	z = z + z_c

	# plot stuff:
	axs[0].clear()
	axs[0].set_xlim(0, plotRange)
	axs[0].set_ylim(0, plotRange)
	axs[0].grid(True)
	plot_r_j = axs[0].plot(sol[:,1],z) # plot r_j
	plot_r_w = axs[0].plot(sol[:,0],z) # plot r_w
	# plt.xlabel('r(z)')
	# plt.ylabel('z')
	axs[0].legend([r"$r_j(z)$",r"$r_w(z)$"])
	
	# plot wind lines
	b = -z0
	m1 = -b/r_in
	m2 = -b/r_out

	focal_point = [0,b]
	windstart = [(1000-b)/m1,1000]
	windend = [(1000-b)/m2,1000]

	x_values = [windstart[0], focal_point[0], windend[0]]
	y_values = [windstart[1], focal_point[1], windend[1]]
	axs[0].plot(x_values, y_values,'--')

	# plot jet lines
	axs[0].plot([0,(z_c-b)/m1], [0,z_c],'--')

	# get all of the other parameters after solving:
	#######################################################################################
	L_w = L_j/s_parameter.val

	# find U_w0 and rho_w0:
	r_w = sol[:,0]
	r_j = sol[:,1]
	alp_j = sol[:,2]
	r_0 = r_w/(1+z/z0)
	th_w = np.arctan(r_w/(z+z0))

	U_w0   = u_win*np.sqrt(r_in/r_0)
	rho_w0 = L_w/(G*M_BH*2*np.pi*r_0*U_w0*np.cos(th_w)*b_w*np.log(r_out/r_in))

	# solve for rho_w:
	sol = opt.root(new_rho_w,[rho_w0],args=(rho_w0,U_w0,p_w0,z,r_w))
	rho_w = sol.x[0]

	# and from it calculate teh rest:
	U_w   = U_w0*rho_w0/rho_w*(1+z/z0)**2
	p_w   = p_w0*np.power(rho_w/rho_w0,gam)

	# use that solution to build a_w,M_w,th_w:
	a_s = np.sqrt(p_w/rho_w)
	M_w = U_w/a_s

	zeta = alp_j-th_w
	del_w = np.arctan((M_w**2-1+2*f1(M_w,zeta)*np.cos((k*np.pi+np.arccos(f2(M_w,zeta)))/3))/(3*(1+(gam-1)/2*M_w**2)*np.tan(zeta)))

	# calculate p_ws and U_ws from del_w and M_w:
	rr = 1/(2/((gam+1)*np.square(M_w*np.sin(del_w)))+(gam-1)/(gam+1))
	U_ws = U_w*np.sin(del_w)/np.sin(del_w-zeta)/rr
	p_ws = (2*gam/(gam+1)*M_w**2*np.sin(del_w)**2-(gam-1)/(gam+1))*p_w
	#######################################################################################

	# plot some stuff
	axs[1].clear()
	axs[2].clear()
	axs[3].clear()

	axs[1].loglog(z,p_ws,label=r'$\rho_{ws}$')
	axs[1].loglog(z,p_w,label=r'$\rho_w$')
	axs[1].loglog(z,z**(-2),'--',label=r'$z^{-2}$',color='k')
	axs[2].plot(z,th_w,label=r'$\theta_w$')
	axs[2].plot(z,alp_j,label=r'$\alpha_j$')
	axs[2].plot(z,zeta,label=r'$\zeta$')
	axs[3].semilogx(z,M_w,label=r'$M_w$')
	axs[3].semilogx(z,rr,label=r'$r$')
	axs[3].semilogx(z,np.sin(del_w),label=r'$\sin (\delta_w)$')

	axs[1].set_xlabel('z')
	axs[2].set_xlabel('z')
	axs[3].set_xlabel('z')
	axs[1].grid(True)
	axs[2].grid(True)
	axs[3].grid(True)
	axs[1].legend()
	axs[2].legend()
	axs[3].legend()

	plt.show()









# visual things--------------------------------------------------------------------
# define plots
fig, axs = plt.subplots(nrows=1,ncols=4,figsize=(18,4.5))
plt.tight_layout()

# define slider for b_w
ax_parameter = plt.axes([0.02, 0.02, 0.9, 0.03])
s_parameter = Slider(ax_parameter, r'$\chi$', 1.0, 10.0 , valstep=1.0, valinit=5.0)
s_parameter.on_changed(update)










# Model Definitions ------------------------------------------------------------------------------
# define del_w functions, since this is just too long
def f1(M_w,zeta):
	returnValue = np.sqrt((M_w**2-1)**2-3*(1+(gam-1)/2*M_w**2)*(1+(gam+1)/2*M_w**2)*np.tan(zeta)**2)
	return returnValue

def f2(M_w,zeta):
	a = ((M_w**2-1)**3-9*(1+(gam-1)/2*M_w**2)*(1+(gam-1)/2*M_w**2+(gam+1)/4*M_w**4)*np.square(np.tan(zeta)))
	b = np.power(f1(M_w,zeta),3)
	return a/b

# base wind solution
def new_rho_w(par, *args):
	rho_w = par
	rho_w0, U_w0, p_w0, z, r_w = args
	r_0 = r_w/(1+z/z0)

	element1 = rho_w0**(-5/3)*rho_w**(5/3)
	element2 = U_w0*rho_w0/(5*p_w0)*(1+z/z0)**2
	element3 = (U_w0**2/(5*p_w0)+2/5*G*M_BH/p_w0*(1/r_w-1/r_0)+1/rho_w0)*rho_w

	return element1+element2-element3

# define the model
def model(VARVAR, z):
	r_w, r_j, alp_j = VARVAR

	# global b_w
	L_w = L_j/s_parameter.val

	# find U_w0 and rho_w0:
	r_0 = r_w/(1+z/z0)
	th_w = np.arctan(r_w/(z+z0))

	U_w0   = u_win*np.sqrt(r_in/r_0) #<-------------------------------------------------------------------------
	rho_w0 = L_w/(2*np.pi*G*M_BH*U_w0*np.cos(th_w)*b_w*np.log(r_out/r_in)*r_0)

	# solve for rho_w:
	sol = opt.root(new_rho_w,rho_w0,args=(rho_w0,U_w0,p_w0,z,r_w))
	rho_w = sol.x[0]

	# and from it calculate the rest:
	U_w   = U_w0*rho_w0/rho_w*(1+z/z0)**2
	p_w   = p_w0*np.power(rho_w/rho_w0,gam)

	# use that solution to build a_w,M_w:
	a_s = np.sqrt(p_w/rho_w)
	M_w = U_w/a_s

	# from them calculate zeta and del_w:
	zeta = alp_j-th_w #<-------------------------------------------------------------------------
	# print((M_w**2-1)**2-3*(1+(gam-1)/2*M_w**2)*(1+(gam+1)/2*M_w**2)*np.tan(zeta)**2)
	del_w = np.arctan((M_w**2-1+2*f1(M_w,zeta)*np.cos((k*np.pi+np.arccos(f2(M_w,zeta)))/3))/(3*(1+(gam-1)/2*M_w**2)*np.tan(zeta)))

	# calculate p_ws and U_ws from del_w and M_w:
	rr = 1/(2/((gam+1)*np.square(M_w*np.sin(del_w)))+(gam-1)/(gam+1))
	U_ws = U_w*np.sin(del_w)/np.sin(del_w-zeta)/rr
	p_ws = (2*gam/(gam+1)*M_w**2*np.sin(del_w)**2-(gam-1)/(gam+1))*p_w

	# and finally find the defivatives of each variable:
	dr_wdz   = np.tan(del_w+th_w)
	dr_jdz   = np.tan(alp_j)
	dalp_jdz = (r_L**2/np.power(r_j,3)-3*np.pi*p_ws*r_j/(2*L_j))*np.square(np.cos(alp_j))

	return [dr_wdz,dr_jdz,dalp_jdz]

plt.show()