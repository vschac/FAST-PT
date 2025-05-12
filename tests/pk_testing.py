#!/usr/bin/env python

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tempfile
import sys

try:
    import camb
    from camb import model
    print(f"CAMB version: {camb.__version__}")
except ImportError:
    print("CAMB is not installed. Please install with:")
    print("pip install camb")
    sys.exit(1)


class CAMBPowerSpectra:
    """Class to generate and compare power spectra using different CAMB approaches"""
    
    def __init__(self, 
                 camb_executable=None,
                 h=0.69, 
                 omega_m=0.3, 
                 omega_b=0.048, 
                 omega_cdm=None,
                 As=2.19e-9, 
                 ns=0.97,
                 mnu=0.0773,
                 num_massive_neutrinos=3,
                 nnu=3.046,
                 omega_k=0.0,
                 tau=0.0697186,
                 w=-1.0):

        self.h = h
        self.H0 = 100 * h
        self.omega_m = omega_m
        self.omega_b = omega_b
        
        self.mnu = mnu
        self.num_massive_neutrinos = num_massive_neutrinos
        self.nnu = nnu
        self.omnuh2 = mnu * num_massive_neutrinos / 93.14
        
        if omega_cdm is None:
            # Neutrino density parameter
            omega_nu = self.omnuh2 / (h**2)
            # Cold dark matter from total matter density
            self.omega_cdm = omega_m - omega_b - omega_nu
        else:
            self.omega_cdm = omega_cdm
        
        # Convert to physical densities
        self.ombh2 = omega_b * (h**2)
        self.omch2 = self.omega_cdm * (h**2)
        
        self.As = As
        self.ns = ns
        self.omega_k = omega_k
        self.tau = tau
        self.w = w
        
        self.camb_executable = self._find_camb_executable(camb_executable)
    
    def _find_camb_executable(self, camb_executable):
        """Find CAMB executable if not specified"""
        if camb_executable is not None:
            if not os.path.isfile(camb_executable):
                print(f"Warning: CAMB executable not found at {camb_executable}")
            return camb_executable
        
        # Look for executable in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        executable_path = os.path.join(script_dir, 'camb')
        
        # Check for Windows extension
        if sys.platform.startswith('win'):
            if os.path.isfile(executable_path + '.exe') and os.access(executable_path + '.exe', os.X_OK):
                return executable_path + '.exe'
        
        # Check for the executable
        if os.path.isfile(executable_path) and os.access(executable_path, os.X_OK):
            return executable_path
        
        print(f"Warning: CAMB executable not found at {executable_path}. INI file approach won't work.")
        return None
    
    def _create_ini_file(self, z=0.0, nonlinear=False, kmax=None, k_per_logint=None):
        """Create CAMB INI file with current parameters"""

        k_per_logint = 50
        
        ini_content = f"""#Parameters for CAMB power spectrum

#output_root is prefixed to output file names
output_root = camb_ps

#What to do
get_scalar_cls = F
get_vector_cls = F
get_tensor_cls = F
get_transfer   = T

# 0: linear, 1: non-linear matter power (HALOFIT)
do_nonlinear = {1 if nonlinear else 0}

#Main cosmological parameters
ombh2          = {self.ombh2}
omch2          = {self.omch2}
omnuh2         = 0.0008308030984886885 #{self.omnuh2}
omk            = {self.omega_k}
hubble         = {self.H0}
mnu            = {self.mnu}

#effective equation of state parameter for dark energy
w              = {self.w}

temp_cmb           = 2.7255
# helium_fraction    = 0.24608761688646366

massless_neutrinos = {self.nnu - self.num_massive_neutrinos}
nu_mass_eigenstates = 1
massive_neutrinos  = {self.num_massive_neutrinos}
share_delta_neff = T
nu_mass_fractions = 1

#Initial power spectrum, amplitude, spectral index and running. Pivot k in Mpc^{{-1}}.
initial_power_num         = 1
pivot_scalar              = 0.05
scalar_amp(1)             = {self.As}
scalar_spectral_index(1)  = {self.ns}
scalar_nrun(1)            = 0

#Reionization
reionization         = T
re_use_optical_depth = T
re_optical_depth     = {self.tau}

do_late_rad_truncation = T

#Which version of Halofit approximation to use
halofit_version = 4
# 1: Smith et al. (2003)
# 2: Bird et al. (2012)
# 3: Original + Cosmology correction
# 4: Takahashi (2012)
# 5: HMcode (2016)
# 6: standard halo model
# 7: PKequal (2016)
# 8: HMcode (2015)
# 9: HMcode (2020) - may not be available in older CAMB versions

#Transfer function settings
transfer_high_precision = T
transfer_kmax           = {kmax}
transfer_k_per_logint   = {k_per_logint}
transfer_num_redshifts  = 1
transfer_interp_matterpower = T
transfer_redshift(1) = {z}
transfer_filename(1)    = transfer_z{z:.1f}.dat
transfer_matterpower(1) = matterpower_z{z:.1f}.dat

#which variable to use for matter power spectrum - 8 is CDM+baryon
transfer_power_var = 8

#Computation parameters
accuracy_boost          = 1.5
l_accuracy_boost        = 1.5
l_sample_boost         = 1.5
"""
        
       
        return ini_content
    
    def generate_power_via_ini(self, k_array, z=0.0, nonlinear=False, save_ini=False):

        if self.camb_executable is None:
            raise ValueError("CAMB executable not found. Cannot use INI file approach.")
        
        # Get the maximum k value from the user's array to ensure full coverage
        kmax = np.max(k_array)# * 1.2  # Add 20% margin for safety
        
        # Create a temporary directory for CAMB files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create INI file
            ini_path = os.path.join(temp_dir, 'params.ini')
            ini_content = self._create_ini_file(z=z, nonlinear=nonlinear, kmax=kmax)
            
            with open(ini_path, 'w') as f:
                f.write(ini_content)
            
            # Save a copy of the INI file for debugging if requested
            if save_ini:
                debug_ini_path = 'camb_debug_params.ini'
                with open(debug_ini_path, 'w') as f:
                    f.write(ini_content)
                print(f"Saved INI file for debugging to: {debug_ini_path}")
            
            # Run CAMB with better error handling
            print(f"Running CAMB with executable: {self.camb_executable}")
            try:
                result = subprocess.run([self.camb_executable, ini_path], 
                                      cwd=temp_dir, 
                                      check=False,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True)
                
                # Check if CAMB ran successfully
                if result.returncode != 0:
                    print("\nCAMB Error Output:")
                    print(result.stderr)
                    print("\nCAMB Standard Output:")
                    print(result.stdout)
                    
                    # Save INI file automatically on error
                    if not save_ini:
                        error_ini_path = 'camb_error_params.ini'
                        with open(error_ini_path, 'w') as f:
                            f.write(ini_content)
                        print(f"Saved problematic INI file to: {error_ini_path}")
                    
                    # Now raise the exception
                    result.check_returncode()
            
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CAMB execution failed: {str(e)}") from e
            
            # Read the power spectrum file
            ps_file = os.path.join(temp_dir, f'camb_ps_matterpower_z{z:.1f}.dat')
            if not os.path.exists(ps_file):
                raise FileNotFoundError(f"CAMB did not generate power spectrum file: {ps_file}")
            
            camb_k, camb_pk = np.loadtxt(ps_file, unpack=True)
            
            # Always interpolate to the user-provided k values
            log_interp = interp1d(np.log(camb_k), np.log(camb_pk), 
                                 bounds_error=False, fill_value='extrapolate')
            log_pk = log_interp(np.log(k_array))
            
            return k_array, np.exp(log_pk)
    
    def _setup_camb_params(self, z=0.0, nonlinear=False, kmax=10.0):

        pars = camb.CAMBparams()
        
        pars.set_cosmology(H0=self.H0, 
                          ombh2=self.ombh2, 
                          omch2=self.omch2, 
                          omk=self.omega_k,
                          mnu=self.mnu, 
                          num_massive_neutrinos=self.num_massive_neutrinos, 
                          nnu=self.nnu)
        
        pars.InitPower.set_params(As=self.As, ns=self.ns, pivot_scalar=0.05)
        
        pars.Reion.set_tau(self.tau)
        
        pars.set_dark_energy(w=self.w)
        
        pars.set_accuracy(AccuracyBoost=1.5, lSampleBoost=1.5, lAccuracyBoost=1.5)
        
        pars.set_matter_power(redshifts=[z], kmax=kmax, k_per_logint=50)

        # Transfer settings
        pars.WantTransfer = True
        pars.Transfer.transfer_high_precision     = True
        pars.Transfer.transfer_kmax               = kmax
        pars.Transfer.transfer_k_per_logint       = 50
        pars.Transfer.transfer_interp_matterpower = True
        pars.Transfer.transfer_num_redshifts      = 1
        pars.Transfer.transfer_redshifts          = [z]
        pars.Transfer.transfer_power_var          = 8

        # Neutrinos
        pars.share_delta_neff      = True
        pars.nu_mass_eigenstates   = 1
        pars.num_massive_neutrinos = self.num_massive_neutrinos
        pars.num_nu_massless       = self.nnu - self.num_massive_neutrinos
        pars.nu_mass_fractions     = [1.0]
        # pars.omnuh2                = self.omnuh2 <<< Setting this parameter explicity screws everything up
        
        if nonlinear:
            # Explicitly set NonLinear_both to make sure both power and transfer functions are nonlinear
            pars.NonLinear = model.NonLinear_both
            pars.NonLinearModel.set_params(halofit_version='takahashi')
            
        print(pars)
        return pars
    
    def generate_power_via_interpolator(self, k_array, z=0.0, nonlinear=False):
        """
        Generate power spectrum using the standard Python API interpolator (Method 1)
        
        Parameters
        ----------
        k_array : array-like
            Array of k values (h/Mpc) to evaluate power spectrum at
        z : float, optional
            Redshift
        nonlinear : bool, optional
            Whether to use nonlinear corrections
        """
        # Get the maximum k value from the user's array to ensure full coverage
        kmax = np.max(k_array)# * 1.2  # Add 20% margin for safety
        
        pars = self._setup_camb_params(z=z, nonlinear=nonlinear, kmax=kmax)
        
        if nonlinear:
            # Set to NonLinear_both to ensure both power spectrum and transfer functions are nonlinear
            pars.NonLinear = model.NonLinear_both
        
        
        PK = camb.get_matter_power_interpolator(pars, 
                                                zmin=z, zmax=z, nz_step=1, 
                                                kmax=kmax,
                                                nonlinear=nonlinear,
                                                var1=8, var2=8)
        
        # Evaluate at requested k values
        pk = PK.P(z, k_array)

        
        return k_array, pk
    
    def generate_power_via_custom_grid(self, k_array, z=0.0, nonlinear=False, fine_grid=True):
        """
        Generate power spectrum using a custom k-grid with get_matter_power_spectrum (Method 2)
        
        Parameters
        ----------
        k_array : array-like
            Array of k values (h/Mpc) to evaluate power spectrum at
        z : float, optional
            Redshift
        nonlinear : bool, optional
            Whether to use nonlinear corrections
        fine_grid : bool, optional
            Whether to use a fine grid for more accurate interpolation
        """
        # Get the maximum k value from the user's array
        kmax = np.max(k_array) * 1.2  # Add 20% margin
        
        # Setup CAMB parameters - this is where nonlinear is set in older versions
        pars = self._setup_camb_params(z=z, nonlinear=nonlinear, kmax=kmax)
        
        # Explicitly set nonlinear model again for certainty
        if nonlinear:
            # Set to NonLinear_both to ensure both power spectrum and transfer functions are nonlinear
            pars.NonLinear = model.NonLinear_both
        
        # Calculate results
        results = camb.get_results(pars)
        
        # Create custom k grid
        min_k = np.min(k_array)
        max_k = np.max(k_array)
        
        # Use a fine grid for more accurate interpolation if requested
        if fine_grid:
            npoints = max(5000, len(k_array) * 5)  # Fine k-grid for accurate interpolation
        else:
            npoints = len(k_array)  # Approximately user's grid
        
        # Get matter power spectrum on our custom grid - older API doesn't accept nonlinear flag
        k_h, z_arr, pk_h = results.get_matter_power_spectrum(minkh=min_k, 
                                                         maxkh=max_k, 
                                                         npoints=npoints)
        
        # Interpolate to exactly match the requested k values
        log_interp = interp1d(np.log(k_h), np.log(pk_h[0]), 
                             bounds_error=False, fill_value='extrapolate')
        return k_array, np.exp(log_interp(np.log(k_array)))
    
    def generate_power_via_direct(self, k_array, z=0.0, nonlinear=False):
        """
        Generate power spectrum directly using the ini file parameter values (Method 3)
        This method attempts to match the INI file behavior as closely as possible
        
        Parameters
        ----------
        k_array : array-like
            Array of k values (h/Mpc) to evaluate power spectrum at
        z : float, optional
            Redshift
        nonlinear : bool, optional
            Whether to use nonlinear corrections
        """
        # Get the maximum k value from the user's array
        kmax = np.max(k_array) * 1.2  # Add 20% margin
        
        # Create a completely fresh CAMBparams object
        pars = camb.CAMBparams()
        
        # Set cosmology
        pars.set_cosmology(H0=self.H0, 
                          ombh2=self.ombh2, 
                          omch2=self.omch2, 
                          omk=self.omega_k,
                          mnu=self.mnu, 
                          num_massive_neutrinos=self.num_massive_neutrinos,
                          nnu=self.nnu)
        
        # Set initial power spectrum
        pars.InitPower.set_params(As=self.As, ns=self.ns, pivot_scalar=0.05)
        
        # Set reionization
        pars.Reion.set_tau(self.tau)
        
        # Set dark energy
        pars.set_dark_energy(w=self.w)
        
        # Critical settings to match INI file
        pars.set_accuracy(AccuracyBoost=1.5, lSampleBoost=1.5, lAccuracyBoost=1.5)
        pars.set_matter_power(redshifts=[z], kmax=kmax, k_per_logint=5)
        
        # Set nonlinear model if requested - this is key for older CAMB versions
        if nonlinear:
            # Explicitly set NonLinear_both to ensure both power spectrum and transfer functions are nonlinear
            pars.NonLinear = model.NonLinear_both
            
            # Match INI file halofit_version=4 (Takahashi)
            if hasattr(pars, 'NonLinearModel') and hasattr(pars.NonLinearModel, 'set_params'):
                try:
                    pars.NonLinearModel.set_params(halofit_version='takahashi')
                except:
                    # Fallback for even older versions
                    pars.NonLinearModel = 'HALOFIT'
            else:
                pars.NonLinearModel = 'HALOFIT'
        
        # Print parameter diagnostics
        print(f"\nDirect Match method - nonlinear setting: {nonlinear}")
        print(f"NonLinear setting: {pars.NonLinear}")
        if hasattr(pars, 'NonLinearModel'):
            print(f"NonLinearModel: {pars.NonLinearModel}")
        
        # Calculate results
        results = camb.get_results(pars)
        
        # Get CAMB's matter power spectrum with same settings as INI file
        # Older API doesn't accept nonlinear flag - it's set in the params
        k_h, z_arr, pk_h = results.get_matter_power_spectrum(minkh=np.min(k_array), 
                                                         maxkh=np.max(k_array), 
                                                         npoints=10000)  # High resolution
        
        # Interpolate to requested k values
        log_interp = interp1d(np.log(k_h), np.log(pk_h[0]), 
                             bounds_error=False, fill_value='extrapolate')
        
        return k_array, np.exp(log_interp(np.log(k_array)))
    
    def compare_all_methods(self, k_array, z=0.0, nonlinear=False, plot=True):
        """
        Compare all power spectrum calculation methods
        
        Parameters
        ----------
        k_array : array-like
            Array of k values (h/Mpc) to evaluate power spectrum at
        z : float, optional
            Redshift
        nonlinear : bool, optional
            Whether to use nonlinear corrections
        plot : bool, optional
            Whether to create a comparison plot
        """
        # Generate power spectra from all methods
        results = {}
        methods = {}
        
        # Try INI file approach
        try:
            results['ini'] = self.generate_power_via_ini(k_array, z, nonlinear)
            methods['ini'] = "INI File Approach"
        except Exception as e:
            print(f"Error in INI method: {e}")
        
        # Try all Python API methods
        try:
            results['interpolator'] = self.generate_power_via_interpolator(k_array, z, nonlinear)
            methods['interpolator'] = "Method 1: Interpolator API"
        except Exception as e:
            print(f"Error in interpolator method: {e}")
        
        try:
            results['custom_grid'] = self.generate_power_via_custom_grid(k_array, z, nonlinear)
            methods['custom_grid'] = "Method 2: Custom Grid"
        except Exception as e:
            print(f"Error in custom grid method: {e}")
        
        try:
            results['direct'] = self.generate_power_via_direct(k_array, z, nonlinear)
            methods['direct'] = "Method 3: Direct Match"
        except Exception as e:
            print(f"Error in direct method: {e}")
        
        # Create comparison plot if requested and INI approach worked
        if plot and 'ini' in results:
            # Get reference power spectrum (INI approach)
            k_ref, pk_ref = results['ini']
            
            # Set up figure with subplots
            n_methods = len(results) - 1  # Excluding INI reference
            fig, axs = plt.subplots(n_methods + 1, 1, figsize=(12, 4 + 3*n_methods), sharex=True)
            
            # First plot: All power spectra
            ax = axs[0]
            for method, (k, pk) in results.items():
                if method == 'ini':
                    ax.loglog(k, pk, label=methods[method], color='black', linewidth=2)
                else:
                    ax.loglog(k, pk, label=methods[method], linestyle='--')
            
            ax.set_ylabel('P(k) [(Mpc/h)³]')
            ax.set_title(f'Matter Power Spectrum (z={z}, {"nonlinear" if nonlinear else "linear"})')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend()
            
            # Remaining plots: Relative differences
            method_list = sorted([m for m in results.keys() if m != 'ini'])
            for i, method in enumerate(method_list):
                ax = axs[i+1]
                k, pk = results[method]
                
                # Calculate relative difference
                rel_diff = np.abs(pk - pk_ref) / pk_ref
                max_rel_diff = np.max(rel_diff) * 100  # Convert to percentage
                
                # Plot relative difference on log-log scale
                ax.loglog(k, rel_diff, color=f'C{i+1}')
                ax.set_ylabel(f'|{method}-INI|/INI')
                ax.grid(True, alpha=0.3, which='both')
                
                # Add maximum difference annotation
                ax.text(0.05, 0.9, f'Max difference: {max_rel_diff:.4f}%', 
                       transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            # Set common x-axis label
            axs[-1].set_xlabel('k [h/Mpc]')
            
            plt.tight_layout()
            plt.savefig('camb_methods_comparison.png', dpi=150)
            plt.show()
        
        return results
    
    # For backward compatibility, keep the original functions
    def generate_power_via_python(self, k_array, z=0.0, nonlinear=False):
        """Original Python API method (same as interpolator)"""
        return self.generate_power_via_interpolator(k_array, z, nonlinear)
    
    def compare_methods(self, k_array, z=0.0, nonlinear=False, plot=True):
        """
        Compare power spectra from INI and Python API approaches
        
        Parameters
        ----------
        k_array : array-like
            Array of k values (h/Mpc) to evaluate power spectrum at - required
        z : float, optional
            Redshift
        nonlinear : bool, optional
            Whether to use nonlinear corrections
        plot : bool, optional
            Whether to create a comparison plot
        """
        # Generate power spectra
        try:
            k_ini, pk_ini = self.generate_power_via_ini(k_array, z, nonlinear)
            ini_works = True
        except Exception as e:
            print(f"Error in INI method: {e}")
            ini_works = False
            k_ini, pk_ini = k_array, np.zeros_like(k_array)
        
        k_py, pk_py = self.generate_power_via_python(k_array, z, nonlinear)
        
        # Create comparison plot if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Main plot: Power spectra
            plt.subplot(211)
            if ini_works:
                plt.loglog(k_ini, pk_ini, label='INI approach', color='blue')
            plt.loglog(k_py, pk_py, label='Python API', color='red', linestyle='--')
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('P(k) [(Mpc/h)³]')
            plt.title(f'Matter Power Spectrum (z={z}, {"nonlinear" if nonlinear else "linear"})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Relative difference plot if both methods worked
            if ini_works:
                plt.subplot(212)
                # Calculate relative difference
                rel_diff = np.abs(pk_ini - pk_py) / pk_ini
                max_rel_diff = np.max(rel_diff) * 100  # Convert to percentage
                
                # Plot relative difference on log-log scale
                plt.loglog(k_ini, rel_diff, color='green')
                plt.xlabel('k [h/Mpc]')
                plt.ylabel('Relative Difference |INI-PY|/INI')
                plt.grid(True, alpha=0.3, which='both')
                plt.title('Relative Difference Between Power Spectra')
                
                # Add maximum difference annotation
                plt.figtext(0.15, 0.3, 
                           f'Maximum relative difference: {max_rel_diff:.4f}%',
                           bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('camb_power_comparison.png', dpi=150)
            plt.show()
        
        if ini_works:
            return k_array, pk_ini, pk_py
        else:
            return k_array, None, pk_py


# Example usage
if __name__ == "__main__":
    print("CAMB Power Spectra Generator - Backwards Compatible")
    print("--------------------------------------------------")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Looking for CAMB executable in: {script_dir}")
    
    # # Create generator with default cosmology
    generator = CAMBPowerSpectra()
    
    # if generator.camb_executable:
    #     print(f"Found CAMB executable: {generator.camb_executable}")
    # else:
    #     print("CAMB executable not found. Only Python API will be available.")
    
    # # User-specified k array (required)
    k_array = np.logspace(-4, 1, 2000)  # k from 10^-4 to 10 h/Mpc
    
    # # Compare all methods for linear power spectrum
    # print("\nComparing all methods for linear power spectrum...")
    # generator.compare_all_methods(k_array, z=0.0, nonlinear=False)
    
    # Compare all methods for nonlinear power spectrum
    print("\nComparing all methods for nonlinear power spectrum...")
    generator.compare_methods(k_array, z=0.0, nonlinear=True)
    
    # Custom cosmology example
    print("\nUsing custom cosmology...")
    try:
        custom_cosmo = CAMBPowerSpectra(
            h=0.69, 
            omega_m=0.3, 
            omega_b=0.048,
            As=2.1e-9,
            ns=0.97
        )
        
        # Compare all methods with custom cosmology
        custom_cosmo.compare_methods(k_array, nonlinear=False)
        
    except Exception as e:
        print(f"Error with custom cosmology: {e}")
        
    print("\nScript completed successfully.")