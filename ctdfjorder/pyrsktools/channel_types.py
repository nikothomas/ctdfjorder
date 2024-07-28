#! /usr/bin/env python3

"""
RBR Generation2/Generation3 channel type information. Used internally by the
library – likely to change over time.
"""

label_prefixes = {
    "acc_00": "acceleration",
    "acc_01": "acceleration",
    "alti00": "distance",
    "asal00": "salinity",
    "atmp00": "temperature",
    "baro00": "barometerpressureperiod",
    "baro01": "barometertemperatureperiod",
    "baro02": "barometerpressure",
    "baro03": "barometertemperature",
    "bpr_00": "period",
    "bpr_01": "period",
    "bpr_02": "pressure",
    "bpr_03": "odotemperature",
    "bpr_04": "period",
    "bpr_05": "period",
    "bpr_06": "pressure",
    "bpr_07": "odotemperature",
    "bpr_08": "pressure",
    "bpr_09": "temperature",
    "cnt_00": "count",
    "cond00": "conductivity",
    "cond01": "conductivity",
    "cond02": "conductivity",
    "cond03": "conductivity",
    "cond04": "conductivity",
    "cond05": "conductivity",
    "cond06": "conductivity",
    "cond07": "conductivity",
    "cond08": "conductivity",
    "cond09": "conductivity",
    "cond10": "conductivity",
    "cond11": "conductivity",
    "cond12": "conductivity",
    "cond13": "conductivity",
    "cond14": "conductivity",
    "cond15": "conductivity",
    "cond16": "conductivity",
    "cond17": "conductivity",
    "cond18": "conductivity",
    "cond19": "conductivity",
    "cond20": "conductivity",
    "cond21": "conductivity",
    "cond22": "conductivity",
    "cond23": "conductivity",
    "cond24": "conductivity",
    "cond25": "conductivity",
    "dden00": "energy",
    "ddox00": "oxygenconcentration",
    "ddox01": "oxygensaturation",
    "ddox02": "oxygenconcentration",
    "ddox03": "oxygenconcentration",
    "doxy00": "oxygensaturation",
    "doxy01": "oxygensaturation",
    "doxy02": "oxygensaturation",
    "doxy03": "oxygensaturation",
    "doxy04": "oxygensaturation",
    "doxy05": "oxygensaturation",
    "doxy06": "oxygensaturation",
    "doxy07": "oxygensaturation",
    "doxy08": "oxygensaturation",
    "doxy09": "oxygensaturation",
    "doxy10": "oxygenconcentration",
    "doxy11": "oxygensaturation",
    "doxy12": "oxygensaturation",
    "doxy13": "oxygensaturation",
    "doxy20": "oxygenconcentration",
    "doxy21": "oxygenconcentration",
    "doxy22": "oxygensaturation",
    "doxy23": "oxygenconcentration",
    "doxy24": "oxygenconcentration",
    "doxy25": "oxygensaturation",
    "doxy26": "oxygensaturation",
    "doxy27": "oxygenconcentration",
    "dpth01": "depth",
    "echo00": "period",
    "echo01": "period",
    "eco_00": "genericflurometer",
    "fluo00": "phycoerythrin",
    "fluo01": "chlorophyll",
    "fluo02": "rhodamine",
    "fluo03": "uv",
    "fluo04": "phycocyanin",
    "fluo05": "phycoerythrin",
    "fluo06": "chlorophyll",
    "fluo07": "rhodamine",
    "fluo08": "uv",
    "fluo09": "phycocyanin",
    "fluo10": "chlorophyll",
    "fluo11": "cdom",
    "fluo12": "crudeoil",
    "fluo13": "cyanobacteria",
    "fluo14": "opticalbrighteners",
    "fluo15": "fluoroescein",
    "fluo16": "rhodamine",
    "fluo17": "refinedfuels",
    "fluo18": "btex",
    "fluo19": "phycocyanin",
    "fluo20": "phycoerythrin",
    "fluo21": "customfluorometer",
    "fluo22": "chlorophyll",
    "fluo23": "cdom",
    "fluo24": "crudeoil",
    "fluo25": "cyanobacteria",
    "fluo26": "opticalbrighteners",
    "fluo27": "fluoroescein",
    "fluo28": "rhodamine",
    "fluo29": "refinedfuels",
    "fluo30": "btex",
    "fluo31": "phycocyanin",
    "fluo32": "phycoerythrin",
    "fluo33": "chlorophyll",
    "fluo34": "cdom",
    "fluo35": "phycoerythrin",
    "fluo36": "rhodamine",
    "irr_00": "irradiance",
    "mag_00": "irradiance",
    "meth00": "methane",
    "opt_01": "amplitude",
    "opt_02": "phase",
    "opt_03": "amplitude",
    "opt_04": "phase",
    "opt_05": "phase",
    "opt_06": "amplitude",
    "opt_07": "phase",
    "opt_08": "amplitude",
    "opt_09": "phase",
    "opt_10": "amplitude",
    "opt_11": "phase",
    "opt_12": "amplitude",
    "opt_13": "phase",
    "opt_14": "phase",
    "orp_00": "orp",
    "orp_01": "orp",
    "par_00": "par",
    "par_01": "par",
    "par_02": "par",
    "par_03": "par",
    "pco200": "partialco2pressure",
    "peri00": "period",
    "peri01": "period",
    "ph__00": "ph",
    "ph__01": "ph",
    "ph__02": "ph",
    "phas00": "phase",
    "pres00": "pressure",
    "pres01": "pressure",
    "pres02": "pressure",
    "pres03": "pressure",
    "pres04": "pressure",
    "pres05": "pressure",
    "pres06": "pressure",
    "pres07": "pressure",
    "pres08": "seapressure",
    "pres09": "pressure",
    "pres10": "pressure",
    "pres11": "pressure",
    "pres12": "pressure",
    "pres13": "pressure",
    "pres14": "pressure",
    "pres15": "pressure",
    "pres16": "pressure",
    "pres17": "pressure",
    "pres18": "pressure",
    "pres19": "pressure",
    "pres20": "pressure",
    "pres21": "pressure",
    "pres22": "pressure",
    "pres23": "codapressure",
    "pres24": "pressure",
    "pres25": "pressure",
    "pres26": "pressure",
    "sal_00": "salinity",
    "sal_01": "salinitycorrected",
    "scon00": "specificconductivity",
    "slop00": "energy",
    "sos_00": "speedofsound",
    "temp00": "temperature",
    "temp01": "rinkotemperature",
    "temp02": "temperature",
    "temp03": "temperature",
    "temp04": "temperature",
    "temp05": "pressuretemperature",
    "temp06": "metstemperature",
    "temp07": "aadioptodetemperature",
    "temp08": "temperature",
    "temp09": "temperature",
    "temp10": "pressuretemperature",
    "temp11": "conductivitycelltemperature",
    "temp12": "temperature",
    "temp13": "codatemperature",
    "temp14": "temperature",
    "temp15": "odotemperature",
    "temp16": "odotemperature",
    "temp17": "odotemperature",
    "temp18": "temperature",
    "temp19": "temperature",
    "temp20": "temperature",
    "temp21": "temperature",
    "temp22": "conductivitycelltemperature",
    "temp23": "obstemperature",
    "temp24": "odotemperature",
    "temp25": "odotemperature",
    "temp26": "legatotemperature",
    "temp27": "legatotemperature",
    "temp28": "pressuretemperature",
    "temp29": "pressuretemperature",
    "temp30": "radtemperature",
    "temp31": "temperature",
    "temp32": "temperature",
    "temp33": "temperature",
    "temp34": "temperature",
    "temp35": "temperature",
    "temp36": "temperature",
    "temp37": "odotemperature",
    "temp38": "temperature",
    "temp39": "tridentetemperature",
    "temp40": "tridentetemperature",
    "tran00": "transmittance",
    "tran01": "transmittance",
    "tran02": "transmittance",
    "turb00": "turbidity",
    "turb01": "turbidity",
    "turb03": "turbidity",
    "turb04": "backscatter",
    "turb05": "turbidity",
    "turb06": "turbidity",
    "turb07": "turbidity",
    "volt00": "voltage",
    "volt01": "voltage",
    "volt02": "voltage",
    "volt03": "voltage",
    "volt04": "voltage",
    "volt05": "voltage",
    "volt06": "voltage",
    "volt07": "voltage",
    "volt99": "voltage",
    "wave00": "height",
    "wave01": "period",
    "wave02": "height",
    "wave03": "period",
    "wave04": "height",
    "wave05": "period",
    "wave06": "height",
    "wave07": "period",
    "wave08": "energy",
}