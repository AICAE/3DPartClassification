
## KiCAD dataset summary

`git clone `


### Categories

                          category  counts
0           Capacitor_SMD.3dshapes      84
1         Connector_Molex.3dshapes      89
2    Connector_Phoenix_MC.3dshapes     180
3                 Crystal.3dshapes      99
4               Diode_THT.3dshapes      89
5            Inductor_SMD.3dshapes      41
6                 LED_THT.3dshapes      74
7             Package_BGA.3dshapes      78
8             Package_DIP.3dshapes     202
9              Package_SO.3dshapes     184
10              Relay_THT.3dshapes      47
11           Resistor_THT.3dshapes     105
12  TerminalBlock_Phoenix.3dshapes      86
13        Transformer_THT.3dshapes      29



```py
kicad_sample_summary = {'Button_Switch_SMD': 101,  'Button_Switch_THT': 63,

 'Capacitor_SMD': 90,  'Capacitor_THT': 375, 

'Connector_Dsub': 112, 'Connector_FFC-FPC': 63,  'Connector_JST': 129,

 'Connector_Molex': 89, 'Connector_Phoenix_GMSTB': 88, 'Connector_Phoenix_MC': 180, 'Connector_Phoenix_MC_HighVoltage': 66, 'Connector_Phoenix_MSTB': 180, 'Connector_PinHeader_1': 278, 'Connector_PinHeader_2': 278, 'Connector_PinSocket_1': 246, 'Connector_PinSocket_2': 278, 'Converter_DCDC': 51, 

'Crystal': 99,

 'Diode_THT': 89, 

'Display_7Segment': 44, 

'Inductor_SMD': 41, 'Inductor_THT': 143, 

'LED_THT': 77, 'Mounting_Wuerth': 217, 

'Package_BGA': 78, 'Package_DFN_QFN': 220, 'Package_DIP': 202, 'Package_QFP': 68, 'Package_SO': 184, 'Package_TO_SOT_SMD': 68, 'Package_TO_SOT_THT': 97, 

'Relay_THT': 48, 'Resistor_THT': 105, 

'TerminalBlock_Phoenix': 86, 'Varistor': 100
}
```

### data conversion not needed, step file format is already there

wrl file format, mesh format

your wrl file is generated from a STEP file by kicad StepUp...
 so you would need to get the source file for this wrl (the STEP and the FreeCAD file).
 https://kicad.github.io/packages3d/Resistor_THT
 edit:
 a related topic at kicad forum
 [https://forum.kicad.info/t/i-created-a- ... brary/5151

[how to create kicad 3D model](https://forum.kicad.info/t/i-created-a-new-kicad3dmodels-library/5151)


### traceparts is not free to batch download

<traceparts.com>



TraceParts Classification
    Mechanical components
        Fasteners
        Bearings
        Brakes, clutches and couplings
        Linear and rotary motion
        Power transmission
        Casters, wheels, handling trolleys
        Jig and fixture construction
        Handles
        Hinges, latches & locks
        Sealing
        Shock/vibration absorbers, springs
        Solenoids, Electromagnets
