import  scipy.constants as spc
import  numpy as np
from    matplotlib import  pyplot as plt
from    typing import Dict, Tuple, Optional

class Task1:
    
    __slots__ = ["E_0", "E_F", "effMass", "channelWidth_y", "channelWidth_z"]

    def __init__(self, 
                 relativeFermiEnergy: float,
                 effectiveMass: float,
                 channelWidth_y: float,
                 groundState: float = 0
                 ) -> None:
        """Sets up the system from task 1. 

        Args:
        relativeFermiEnergy (float): How many electron volts the fermi 
            energy is above the ground state in the z-direction.
        effectiveMass (float): Effective mass of an electron in the system as a percentage of
            the electron mass of a free electron. 
        channelWidth (float): The width of the channel in the y-direction.
        groundState (float, optional): The ground state energy in eV in the z-direction. Defaults to 0 
            for reference.
        """
                
        # Effective mass of an electron in the system in kg
        self.effMass = effectiveMass * spc.electron_mass
        
        # Channel dimension and ground state energu in the z-direction. Since the wavefunction in the
        # z-direction is restricted to the ground state, the 2. state energy eigenvalue
        # must be below the fermi energy.
        self.channelWidth_z, self.E_0 = self.maxChannelWidth_z()

        # Fermi energy is 90 meV above ground 
        # state energy in z-direction.
        self.E_F = self.E_0 + relativeFermiEnergy
        
        # Width of the channel in y-direction in meters
        self.channelWidth_y = channelWidth_y
        
    def maxChannelWidth_z(self, plot: bool = False) -> Tuple[float, float]:
        """Plots the energy eigenvalue of the first excited state in the z-direction.
        Meant to find out ca. the maximum channel width in the z-direction where psi(z) is
        still truly one dimensional.

        Args:
            channelWidth_z (np.ndarray): List of channel widths to be plotted
        Returns:
            Tuple with max channel width as first index and energy eigenvalue of the 
            ground state as second index.
        """
        
        widths = np.linspace(10e-9, 20e-9, 10000)
        E_z =   ( 
                 ( np.pi**2 * (spc.hbar**2)) 
                 / 
                 ( 2 * self.effMass * widths**2 * spc.electron_volt ) 
                ) * 10**3
        
        y = {
            "Ground state" : E_z,
            "First excited state" : E_z * 2**2,
            "Fermi energy" : E_z + 90
            }
        idx = np.abs(y["First excited state"] - (y["Ground state"] + 90)).argmin()
        if y["First excited state"][idx] < 90:
            idx -= 1
        maxWidth = widths[idx]
        
        if plot:
            self._plot(
                widths, 
                y, 
                "$E_{z_0}$ and $E_{z_1}$ vs. channel width in z-direction.",
                "Channel width [m]",
                "Energy [meV]")
            plt.axvline(maxWidth, color="green", linestyle="dashed", label="Max. channel width")
            
            lines = plt.gca().get_lines()  # Get all the lines in the current plot
            for line in lines:
                if line.get_label() == 'Fermi energy':  # Check if the label matches
                    line.set_linestyle('dashed')  # Change the linestyle
                    line.set_color("red")
            plt.xlim(left=1e-8)
            plt.gca().ticklabel_format(style="scientific", axis="both", scilimits=(0,0))
            plt.legend(fontsize=20)
            plt.show()
                
        return maxWidth, E_z[idx]/10**3
        
    def E_x(self, k_x: np.array) -> np.array:
        """
        Energy eigenvalues for psi(x) as a function of k_x.
        """
        return ( 
                ( (spc.hbar**2) * k_x**2 ) 
                / 
                ( 2 * self.effMass * spc.electron_volt) 
               )
    
    def E_y(self, n: int) -> float:
        """
        Energy eigenvalues of psi(y) as a function of
        the quantum number n.
        """
        return  ( 
                 ( np.pi**2 * (spc.hbar**2) * n**2 ) 
                 / 
                 ( 2 * self.effMass * self.channelWidth_y**2 * spc.electron_volt) 
                )
        
    def E_n(self, k_k: np.array, n_min: int, n_max: int) -> Dict:
        """
        Returns a dictionary with the energy bands of the system.
        """
        
        bands = {}
        
        for n in range(n_min, n_max + 1):
            bands[f"n = {n}"] = ( self.E_0 + self.E_x(k_k) + self.E_y(n) ) 
            
        return bands
    
    def plotEnergyBands(self, band_min: int, band_max: int) -> None:
        """Plots the energy bands of the system.

        Args:
            band_min (int): The lowest band to be included.
            band_max (int): The highest band to be included.
        """

        k_x: np.ndarray = np.linspace(-1e9, 1e9, 1000)
        bands: dict = self.E_n(k_x, band_min, band_max)
        
        closestIndex = min(range( len(bands["n = 1"]) ), key=lambda i: abs(bands["n = 1"][i]
            - (self.E_F + 30e-3) ))  
        x_max = k_x[closestIndex]
        
        self._plot(k_x, bands, "Energy bands of 2DEG within the constriction", "$k_x$ [$m^{-1}$]", "Energy [eV]")
        
        plt.axhline(self.E_F, linestyle="dashed", color="red", label="$E_F$")
        plt.axhline(self.E_0, linestyle="dashdot", color="green", label="$E_{z_0}$")
        plt.ylim(top= self.E_F + 30e-3, bottom = 0 )
        plt.xlim(left = -x_max, right=x_max)
        plt.legend(fontsize=15)
        
        plt.show()

    def _plot(self, 
              xVal: np.array, 
              yVals: dict, 
              title: str, 
              xLabel: str, 
              yLabel: str, 
              figsize: Tuple[int, int] = (10,6),
              savePath: Optional[str]  = None
              ) -> None:

        font = {'family': 'serif', 'color': 'darkred', 
                'weight': 'normal', 'size': 16,}

        # Create the plot
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=35, fontdict=font, y=1.05)
        plt.xlabel(xLabel, fontsize=30, fontdict=font)
        plt.ylabel(yLabel, fontsize=30, fontdict=font)

        for key in yVals:
            plt.plot(xVal, yVals[key], label=key)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.gca().ticklabel_format(style="sci", axis="both", scilimits=(0,0))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().xaxis.get_offset_text().set_fontsize(18)
        plt.gca().yaxis.get_offset_text().set_fontsize(18)
        
        if savePath is not None:
            plt.savefig(savePath)

    def bandsBelowFermiEnergy(self) -> int:
        """
        Returns an integer representing the nr. of bands
        below the fermi energy in the system.
        """
        
        n = 1
        energy = 0
        while energy < self.E_F:
            
            bandMinimum = self.E_0 + self.E_y(n)
            if bandMinimum <= self.E_F:
                n += 1
            else:
                return n -1

    def getCurrent(self, V_sd: float) -> float:
        """
        Calculates the current through the channel based on the applied
        voltage between the source- and drain terminals.
        """
        current = (
            self.bandsBelowFermiEnergy() * 
            ( (2 * spc.elementary_charge**2) / spc.h ) *
            V_sd
        )
        
        return current
          
if __name__ == "__main__":     
    
    system = Task1(
        relativeFermiEnergy = 90e-3,
        effectiveMass = 0.097,
        channelWidth_y = 120e-9,
        groundState = 0
    )

    #nrBands = system.bandsBelowFermiEnergy()
    #print(f"Maximum width in z-direction for psi(z)" 
    #      f"to be constricted to the ground state: {system.maxChannelWidth_z(plot=True)[0]*10**9} nm")
    #print(f"Bands below the fermi energy: {nrBands}")
    #print(f"Energy of band nr. {nrBands}: {system.E_0 + system.E_y(nrBands)} eV")
    print(f"Current as a result of V_sd = 30 uV: {system.getCurrent(V_sd = 30e-6)*1e9} nA")
    #system.plotEnergyBands(band_min = 1, band_max = 19)
